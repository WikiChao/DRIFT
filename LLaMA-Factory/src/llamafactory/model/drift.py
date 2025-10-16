import re
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from typing import Dict, List, Optional, Tuple

# -------------------------------
# What gets nudged (name patterns)
# -------------------------------
MERGE_PATTERNS = [
    r"\.mlp\.(gate_proj|up_proj|down_proj)$",
    r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
    # r"\.input_layernorm$", r"\.post_attention_layernorm$",
    # r"\.embed_tokens$",  r"lm_head$",
]

# NEW: simple tokens you can pass instead of raw regex
MERGE_PRESETS = {
    "mlp": [r"\.mlp\.(gate_proj|up_proj|down_proj)$"],
    "attn": [r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"],
    "norm": [r"\.input_layernorm$", r"\.post_attention_layernorm$"],
    "embed": [r"\.embed_tokens$"],
    "head": [r"lm_head$"],
    # convenience bundles
    "default": [
        r"\.mlp\.(gate_proj|up_proj|down_proj)$",
        r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
    ],
    "full": [
        r"\.mlp\.(gate_proj|up_proj|down_proj)$",
        r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
        r"\.input_layernorm$", r"\.post_attention_layernorm$",
        r"\.embed_tokens$", r"lm_head$",
    ],
}

def _matches(name: str) -> bool:
    return any(re.search(p, name) for p in MERGE_PATTERNS)

# NEW: expand tokens like "mlp", "attn", "default" into regex patterns
def _expand_merge_patterns(spec: Optional[List[str]]) -> List[str]:
    if spec is None:
        return MERGE_PATTERNS
    out: List[str] = []
    for item in spec:
        key = (item or "").strip()
        if key in MERGE_PRESETS:
            out.extend(MERGE_PRESETS[key])
        elif key.startswith("re:"):
            out.append(key[3:])
        else:
            # treat as regex as-is
            out.append(key)
    return out

# ---------------------------------------
# Per-pattern alpha overrides (optional)
# ---------------------------------------
MERGE_CFG = dict(
    default_alpha=1e-0,   # global fallback if no explicit alpha passed to DeltaDirectionPrior
    overrides={
        # Example:
        # r"\.mlp\.(gate_proj|up_proj|down_proj)$": 0.02,
    },
)

def _alpha_for(name: str, global_alpha: Optional[float]):
    if global_alpha is not None:
        return float(global_alpha)
    for pat, a in MERGE_CFG["overrides"].items():
        if re.search(pat, name):
            return float(a)
    return float(MERGE_CFG["default_alpha"])

_LAYER_INDEX_PATTERNS = [r"\.layers\.(\d+)\.", r"\.h\.(\d+)\."]

def _get_layer_idx(full_name: str):
    for pat in _LAYER_INDEX_PATTERNS:
        m = re.search(pat, full_name)
        if m:
            return int(m.group(1))
    return None

# -------------------------------
# Helpers
# -------------------------------
@torch.no_grad()
def _l2_norm(x: torch.Tensor) -> torch.Tensor:
    return x.norm().clamp_min(1e-12)

def _get_by_path(root: nn.Module, dotted: str) -> nn.Module:
    m = root
    for part in dotted.split("."):
        m = getattr(m, part)
    return m

def _is_rms_or_layer_norm(m: nn.Module) -> bool:
    return isinstance(m, nn.LayerNorm) or m.__class__.__name__.endswith("RMSNorm")

# ============================================================
# Gradient-time prior: push updates along Δ
#   - single:   Δ = (prior1 - sft)
#   - multiple: Δ = (prior2 - prior1)
# ============================================================
class DeltaDirectionPrior:
    """
    Build once, then call .attach() before training to register grad hooks.
    No parameters are modified up front; only grads are adjusted during backward.

    Modes:
    - scale="absolute": grad <- grad - α * Δ
    - scale="gradnorm" (default): grad <- grad - α * ||grad|| * Δ_hat
    - scale="adaptive_cosine": grad <- grad - α' * ||grad|| * Δ_hat, where α' = α * (cos(grad,Δ)+1)/2
    """

    def __init__(
        self,
        sft_model: PreTrainedModel,                   # the model you're fine-tuning (current weights)
        prior1_model: Optional[PreTrainedModel] = None,  # first prior (single mode when provided alone)
        prior2_model: Optional[PreTrainedModel] = None,  # second prior (multiple mode: Δ = prior2 - prior1)
        *,
        alpha: Optional[float] = None,    # global α; per-pattern overrides in MERGE_CFG still apply
        skip_first_n: int = 0,
        only_layers: Optional[List[int]] = None,
        normalize_delta_to_param: bool = True,
        include_bias: bool = True,
        scale: str = "gradnorm",        # "gradnorm" or "absolute"
        eps: float = 1e-12,
        merge_patterns: Optional[List[str]] = None,  # NEW: override which module names to match
    ):
        self.alpha_global = alpha
        self.skip_first_n = int(skip_first_n)
        self.only_layers = set(only_layers) if only_layers is not None else None
        self.normalize_delta_to_param = bool(normalize_delta_to_param)
        self.include_bias = bool(include_bias)
        self.scale = scale
        self.eps = eps
        # CHANGED: allow simple names or raw regex
        self.merge_patterns = _expand_merge_patterns(merge_patterns)

        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._param_to_delta: Dict[nn.Parameter, torch.Tensor] = {}
        self._param_to_alpha: Dict[nn.Parameter, float] = {}

        # Precompute deltas module-by-module to respect pattern filtering
        if (prior1_model is not None) and (prior2_model is not None):
            # multiple: Δ = prior2 - prior1, attach to sft_model params
            self._build_deltas_from_priors(sft_model, prior1_model, prior2_model)
            print("[delta-prior] multiple: Δ = prior2 - prior1")
        elif prior1_model is not None:
            # single: Δ = prior1 - sft_model
            self._build_deltas_prior_minus_sft(sft_model, prior1_model)
            print("[delta-prior] single: Δ = prior - sft")
        else:
            raise ValueError("DeltaDirectionPrior requires prior1_model (single) or prior1_model+prior2_model (multiple).")

    # NEW: instance-level matcher using provided patterns
    def _name_matches(self, name: str) -> bool:
        return any(re.search(p, name) for p in self.merge_patterns)

    @torch.no_grad()
    def _iter_filtered_modules(self, sft_model: PreTrainedModel):
        for full_name, mod_sft in sft_model.named_modules():
            # skip visual or non-matching modules
            if "visual" in full_name or not self._name_matches(full_name):  # CHANGED: use instance patterns
                continue

            layer_idx = _get_layer_idx(full_name)
            if layer_idx is not None:
                if layer_idx < self.skip_first_n:
                    continue
                if self.only_layers is not None and layer_idx not in self.only_layers:
                    continue

            yield full_name, mod_sft

    @torch.no_grad()
    def _build_deltas_prior_minus_sft(self, sft_model: PreTrainedModel, prior_model: PreTrainedModel):
        # Δ = prior - sft (single)
        for full_name, mod_sft in self._iter_filtered_modules(sft_model):
            try:
                mod_prior = _get_by_path(prior_model, full_name)
            except AttributeError:
                continue  # structure mismatch, ignore

            alpha_m = _alpha_for(full_name, self.alpha_global)
            if alpha_m == 0.0:
                continue

            if isinstance(mod_sft, nn.Linear) and isinstance(mod_prior, nn.Linear):
                self._register_param_delta(
                    mod_sft.weight, mod_prior.weight, alpha_m, full_name + ".weight"
                )
                if self.include_bias and (mod_sft.bias is not None):
                    b2 = getattr(mod_prior, "bias", None)
                    if b2 is None:
                        z = torch.zeros_like(mod_sft.bias)
                        self._register_param_delta(mod_sft.bias, z, alpha_m, full_name + ".bias")
                    else:
                        self._register_param_delta(mod_sft.bias, b2, alpha_m, full_name + ".bias")

            elif isinstance(mod_sft, nn.Embedding) and isinstance(mod_prior, nn.Embedding):
                self._register_param_delta(mod_sft.weight, mod_prior.weight, alpha_m, full_name + ".weight")

            elif _is_rms_or_layer_norm(mod_sft) and _is_rms_or_layer_norm(mod_prior):
                if getattr(mod_sft, "weight", None) is not None and getattr(mod_prior, "weight", None) is not None:
                    self._register_param_delta(mod_sft.weight, mod_prior.weight, alpha_m, full_name + ".weight")
                if self.include_bias and getattr(mod_sft, "bias", None) is not None:
                    b2 = getattr(mod_prior, "bias", None)
                    if b2 is None:
                        z = torch.zeros_like(mod_sft.bias)
                        self._register_param_delta(mod_sft.bias, z, alpha_m, full_name + ".bias")
                    else:
                        self._register_param_delta(mod_sft.bias, b2, alpha_m, full_name + ".bias")

    @torch.no_grad()
    def _build_deltas_from_priors(
        self,
        sft_model: PreTrainedModel,
        prior1_model: PreTrainedModel,
        prior2_model: PreTrainedModel,
    ):
        # Δ = prior2 - prior1 (multiple); attach to sft_model params
        for full_name, mod_sft in self._iter_filtered_modules(sft_model):
            try:
                mod_p1 = _get_by_path(prior1_model, full_name)
                mod_p2 = _get_by_path(prior2_model, full_name)
            except AttributeError:
                continue  # structure mismatch, ignore

            alpha_m = _alpha_for(full_name, self.alpha_global)
            if alpha_m == 0.0:
                continue

            if isinstance(mod_sft, nn.Linear) and isinstance(mod_p1, nn.Linear) and isinstance(mod_p2, nn.Linear):
                # weights
                if mod_sft.weight.shape == mod_p1.weight.shape == mod_p2.weight.shape:
                    self._register_param_delta_from_priors(
                        mod_sft.weight, mod_p1.weight, mod_p2.weight, alpha_m, full_name + ".weight"
                    )
                # bias
                if self.include_bias and (mod_sft.bias is not None):
                    b1 = getattr(mod_p1, "bias", None)
                    b2 = getattr(mod_p2, "bias", None)
                    if b1 is None:
                        b1 = torch.zeros_like(mod_sft.bias)
                    if b2 is None:
                        b2 = torch.zeros_like(mod_sft.bias)
                    if b1.shape == mod_sft.bias.shape and b2.shape == mod_sft.bias.shape:
                        self._register_param_delta_from_priors(
                            mod_sft.bias, b1, b2, alpha_m, full_name + ".bias"
                        )

            elif isinstance(mod_sft, nn.Embedding) and isinstance(mod_p1, nn.Embedding) and isinstance(mod_p2, nn.Embedding):
                if mod_sft.weight.shape == mod_p1.weight.shape == mod_p2.weight.shape:
                    self._register_param_delta_from_priors(
                        mod_sft.weight, mod_p1.weight, mod_p2.weight, alpha_m, full_name + ".weight"
                    )

            elif _is_rms_or_layer_norm(mod_sft) and _is_rms_or_layer_norm(mod_p1) and _is_rms_or_layer_norm(mod_p2):
                # weight
                if getattr(mod_sft, "weight", None) is not None and getattr(mod_p1, "weight", None) is not None and getattr(mod_p2, "weight", None) is not None:
                    if mod_sft.weight.shape == mod_p1.weight.shape == mod_p2.weight.shape:
                        self._register_param_delta_from_priors(
                            mod_sft.weight, mod_p1.weight, mod_p2.weight, alpha_m, full_name + ".weight"
                        )
                # bias
                if self.include_bias and getattr(mod_sft, "bias", None) is not None:
                    b1 = getattr(mod_p1, "bias", None)
                    b2 = getattr(mod_p2, "bias", None)
                    if b1 is None:
                        b1 = torch.zeros_like(mod_sft.bias)
                    if b2 is None:
                        b2 = torch.zeros_like(mod_sft.bias)
                    if b1.shape == mod_sft.bias.shape and b2.shape == mod_sft.bias.shape:
                        self._register_param_delta_from_priors(
                            mod_sft.bias, b1, b2, alpha_m, full_name + ".bias"
                        )
            else:
                # not a handled type; ignore silently
                pass

    @torch.no_grad()
    def _register_param_delta(
        self, p_sft: torch.Tensor, p_prior_like: torch.Tensor, alpha_m: float, tag: str
    ):
        # Δ = prior - sft (single)
        p_sft_cpu = p_sft.detach().to("cpu")
        p_prior_cpu = p_prior_like.detach().to("cpu")

        delta_cpu = (p_prior_cpu - p_sft_cpu)

        if self.normalize_delta_to_param:
            n_delta = delta_cpu.norm().clamp_min(self.eps)
            n_ref = p_sft_cpu.norm().clamp_min(self.eps)
            delta_cpu = delta_cpu * (n_ref / n_delta)

        delta_cpu = delta_cpu.contiguous()

        self._param_to_delta[p_sft] = delta_cpu
        self._param_to_alpha[p_sft] = float(alpha_m)
        print(f"[delta-prior] prepared Δ (single, CPU) for {tag}, alpha={alpha_m:.6f}")

    @torch.no_grad()
    def _register_param_delta_from_priors(
        self, p_target_sft: torch.Tensor, p1_like: torch.Tensor, p2_like: torch.Tensor, alpha_m: float, tag: str
    ):
        # Δ = prior2 - prior1 (multiple), normalized to target (sft) param if requested
        p_target_cpu = p_target_sft.detach().to("cpu")
        p1_cpu = p1_like.detach().to("cpu")
        p2_cpu = p2_like.detach().to("cpu")

        if p1_cpu.shape != p2_cpu.shape or p1_cpu.shape != p_target_cpu.shape:
            # shape mismatch; skip
            return

        delta_cpu = (p2_cpu - p1_cpu)

        if self.normalize_delta_to_param:
            n_delta = delta_cpu.norm().clamp_min(self.eps)
            n_ref = p_target_cpu.norm().clamp_min(self.eps)
            delta_cpu = delta_cpu * (n_ref / n_delta)

        delta_cpu = delta_cpu.contiguous()

        # Store by the target SFT parameter (this is what we attach hooks to)
        self._param_to_delta[p_target_sft] = delta_cpu
        self._param_to_alpha[p_target_sft] = float(alpha_m)
        print(f"[delta-prior] prepared Δ (multiple, CPU) for {tag}, alpha={alpha_m:.6f}")

    def _make_hook(self, p: nn.Parameter):
        delta_cpu = self._param_to_delta[p]
        alpha = self._param_to_alpha[p]
        eps = self.eps
        scale_mode = self.scale

        def _hook(grad: torch.Tensor):
            if grad is None:
                return None

            # Move Δ to device for this grad only (no persistent cache)
            delta_dev = delta_cpu.to(device=grad.device, dtype=grad.dtype, non_blocking=True)

            if scale_mode == "absolute":
                # g' = g - α * Δ (R1 - VL)
                grad = grad - alpha * delta_dev
            elif scale_mode == "adaptive_cosine":
                # Dynamic alpha based on cosine similarity: α' = α * (cos + 1)/2
                g_norm = grad.norm().clamp_min(eps)
                d_norm = delta_dev.norm().clamp_min(eps)

                if (g_norm < eps) or (d_norm < eps):
                    cos_theta = grad.new_tensor(0.0)
                else:
                    cos_theta = torch.dot(grad.flatten(), delta_dev.flatten()) / (g_norm * d_norm)
                    cos_theta = cos_theta.clamp(-1.0, 1.0)

                scaling_factor = 0.5 * (cos_theta + 1.0)  # map [-1,1] -> [0,1]
                # scaling_factor = (1 - cos_theta)
                alpha_dynamic = alpha * scaling_factor

                # g' = g - α' * ||g|| * Δ_hat
                grad = grad - alpha_dynamic * g_norm * (delta_dev / d_norm)
            else:
                # g' = g - α * ||g|| * Δ_hat
                g_norm = grad.norm().clamp_min(eps)
                d_norm = delta_dev.norm().clamp_min(eps)
                grad = grad - alpha * g_norm * (delta_dev / d_norm)

            return grad

        return _hook

    def attach(self, model: PreTrainedModel):
        """
        Register gradient hooks onto the exact Parameter objects of `model`.
        Call once before training. Use .detach() to remove.
        """
        param_set = {p for p in model.parameters()}
        attached = 0
        for p_like, delta in list(self._param_to_delta.items()):
            p = p_like
            if p not in param_set:
                # rare case: try to match by data pointer/size
                matched = None
                for q in param_set:
                    if q.data_ptr() == p_like.data_ptr() and q.shape == p_like.shape:
                        matched = q
                        break
                if matched is None:
                    self._param_to_delta.pop(p_like, None)
                    self._param_to_alpha.pop(p_like, None)
                    continue
                p = matched

            h = p.register_hook(self._make_hook(p))
            self._handles.append(h)
            attached += 1

        print(f"[delta-prior] attached {attached} gradient hooks.")

    def detach(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        print("[delta-prior] detached gradient hooks.")

# --------------------------------------------------------
# Public API: create and attach gradient prior
# --------------------------------------------------------
def attach_delta_prior(
    sft_model: PreTrainedModel,
    prior1_model: Optional[PreTrainedModel] = None,
    prior2_model: Optional[PreTrainedModel] = None,
    *,
    alpha: Optional[float] = None,
    skip_first_n: int = 0,
    only_layers: Optional[List[int]] = None,
    normalize_delta_to_param: bool = True,
    include_bias: bool = True,
    scale: str = "gradnorm",
    merge_patterns: Optional[List[str]] = None,
) -> DeltaDirectionPrior:
    """
    Prepare Δ and register gradient hooks that nudge updates toward Δ during SFT.
      - multiple (prior1_model and prior2_model provided): Δ = prior2 - prior1.
      - single (only prior1_model provided): Δ = prior1 - sft_model.
    merge_patterns accepts simple names (e.g., ["default"], ["mlp","attn"], ["full"])
    or regex strings; prefix with "re:" to force literal regex.
    """
    prior = DeltaDirectionPrior(
        sft_model,
        prior1_model=prior1_model,
        prior2_model=prior2_model,
        alpha=alpha,
        skip_first_n=skip_first_n,
        only_layers=only_layers,
        normalize_delta_to_param=normalize_delta_to_param,
        include_bias=include_bias,
        scale=scale,
        merge_patterns=merge_patterns,
    )
    prior.attach(sft_model)
    return prior