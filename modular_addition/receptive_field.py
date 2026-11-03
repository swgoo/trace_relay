"""Local dependency reach with phase-inheritance edges removed."""

def _windows(values, name):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f'{name} must be a nonempty layer list')
    if any(type(x) is not int or x < 0 for x in values):
        raise ValueError(f'{name} values must be nonnegative integers')
    return list(values)


def direct_receptive_field(read_windows, write_windows, relay_enabled=True, *, delta_windows=None):
    """Propagate sets of source offsets through each actual local dependency.

    An offset d denotes input[t-d]. The theta[t-stride] edge alone is excluded;
    theta[t] still depends on write[t], optionally corrected using past raw
    writes via delta attention. Residuals and Q/K/V dependencies are
    unions, normalization and FFN are pointwise. The resulting width includes t.
    """
    reads = _windows(read_windows, 'read_windows')
    writes = _windows(write_windows, 'write_windows')
    deltas = [0] * len(reads) if delta_windows is None else _windows(delta_windows, 'delta_windows')
    if len(reads) != len(writes) or len(reads) != len(deltas) or type(relay_enabled) is not bool:
        raise ValueError('Matching layer counts and a boolean relay flag required')
    hidden = {0}
    layers = []
    for index, (rw, ww, dw) in enumerate(zip(reads, writes, deltas)):
        write = {d + offset for d in hidden for offset in range(ww + 1)} if relay_enabled else set()
        corrected_write = {d + offset for d in write for offset in range(dw + 1)}
        memory = hidden | corrected_write
        read = {d + offset for d in memory for offset in range(rw + 1)}
        hidden = memory | read
        layers.append(dict(layer=index, read_window=rw, write_window=ww,
                           write_radius=max(write) if write else None,
                           memory_radius=max(memory), output_radius=max(hidden)))
        if relay_enabled and any(deltas):
            layers[-1].update(delta_window=dw, corrected_write_radius=max(corrected_write))
    return dict(width=max(hidden) + 1, radius=max(hidden), layers=layers,
                reachable_offsets=sorted(hidden),
                method=('dependency_offset_union_without_phase_inheritance_delta_v2'
                        if relay_enabled and any(deltas) else
                        'dependency_offset_union_without_phase_inheritance_v1'),
                window_convention='past_tokens_plus_current',
                excluded_edge='phase[t-(write_window+1)] -> phase[t]')
