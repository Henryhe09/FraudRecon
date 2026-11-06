"""Public task names mapped to the original execution modes."""
EVAL_MODE_ALIASES = {
    'full': 'full',
    'without_phases': 'full_no_t2',
    'dialogue_only': 'round1_only',
    'detection_only': 'acc_only',
    'category_only': 'category_only',
    'signals_only': 'signals_only',
    'reasoning_only': 'cot_only',
    'phase_only': 't2_only',
}
LEGACY_MODES = tuple(dict.fromkeys(EVAL_MODE_ALIASES.values()))
CLI_MODES = tuple(dict.fromkeys((*EVAL_MODE_ALIASES, *LEGACY_MODES)))
