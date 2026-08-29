def _apply_compatibility_patches():
    try:
        import transformers.quantizers.base as qbase
        import transformers.modeling_utils as mutils

        if not hasattr(qbase.get_keys_to_not_convert, "_patched"):
            _orig = qbase.get_keys_to_not_convert
            def _safe_get_keys(model):
                if not hasattr(model, "all_tied_weights_keys"):
                    model.all_tied_weights_keys = dict()
                return _orig(model)
            _safe_get_keys._patched = True
            qbase.get_keys_to_not_convert = _safe_get_keys

        if not hasattr(mutils.PreTrainedModel._finalize_model_loading, "_patched"):
            _orig_finalize = mutils.PreTrainedModel._finalize_model_loading
            def _safe_finalize(model, load_config, loading_info):
                if not hasattr(model, "all_tied_weights_keys"):
                    model.all_tied_weights_keys = dict()
                _orig_tie = model.tie_weights
                def _safe_tie(**kwargs):
                    try:
                        return _orig_tie(**kwargs)
                    except TypeError:
                        return _orig_tie()
                model.tie_weights = _safe_tie
                return _orig_finalize(model, load_config, loading_info)
            _safe_finalize._patched = True
            mutils.PreTrainedModel._finalize_model_loading = staticmethod(_safe_finalize)
    except Exception as e:
        pass  # silently skip if transformers version doesn't need patches


_apply_compatibility_patches()