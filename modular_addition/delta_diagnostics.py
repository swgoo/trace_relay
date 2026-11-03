"""Observational ON/OFF probes; no parameter updates or training RNG consumption."""
import math

from .evaluate import evaluate_model
from .utils import capture_rng, restore_rng


class CorrectionStatistics:
    """Raw FP32 write angles and bounded corrections on unpadded task inputs."""
    def __init__(self, model):
        self.model, self.handles, self.sums = model, [], {}

    def _observe(self, key, tensor):
        value = tensor.detach().double()
        entry = self.sums.setdefault(key, dict(count=0, squared_sum=0., absolute_sum=0., max_abs=0.))
        entry['count'] += value.numel()
        entry['squared_sum'] += value.square().sum().item()
        entry['absolute_sum'] += value.abs().sum().item()
        entry['max_abs'] = max(entry['max_abs'], value.abs().max().item())

    def __enter__(self):
        for index, layer in enumerate(self.model.backbone.layers):
            if layer.delta_attention is None:
                continue
            def raw_hook(module, args, output, i=index, scale=layer.scale):
                self._observe((i,'raw_delta'), scale * output.float().tanh())
            def correction_hook(module, args, output, i=index):
                mask = args[5][:,:,None]
                self._observe((i,'correction'),
                    self.model.backbone.config.phase_correction_scale * output.float().tanh() * mask)
            self.handles.extend([layer.write_phase.register_forward_hook(raw_hook),
                                 layer.delta_attention.register_forward_hook(correction_hook)])
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()

    def summary(self):
        result = {}
        for (index,kind), values in self.sums.items():
            result.setdefault(str(index+1),{})[kind] = dict(
                count=values['count'], rms=math.sqrt(values['squared_sum']/values['count']),
                mean_abs=values['absolute_sum']/values['count'],max_abs=values['max_abs'])
        for values in result.values():
            raw = values['raw_delta']['rms']
            values['correction_to_raw_rms_ratio'] = values['correction']['rms']/raw if raw else None
        return result


def evaluate_delta_diagnostics(model, sources, *, batch_size=64,device='cpu',precision='fp32'):
    """Same held-out inputs at scheduled scale and zero scale, with RNG restored.

    ON is the current scheduled scale, including zero at the end of a decay.
    OFF-minus-ON CE is positive when the correction helps. Probes never select
    checkpoints or modify training. Ordinary task inputs have no padding.
    """
    rng, training = capture_rng(), model.training
    scale = model.backbone.config.phase_correction_scale
    result = dict(format='delta-learning-diagnostics-v1',runtime_phase_correction_scale=scale,
                  delta_detach_inputs=model.backbone.config.delta_detach_inputs,
                  used_for_training_control=False,lengths={})
    try:
        for length, entry in sources.items():
            source = entry['ordinary']
            with CorrectionStatistics(model) as stats:
                on = evaluate_model(model,source,r_direct=model.effective_config()['direct_receptive_field']['width'],
                    batch_size=batch_size,device=device,precision=precision)
            with model.temporary_phase_correction_scale(0.):
                off = evaluate_model(model,source,r_direct=model.effective_config()['direct_receptive_field']['width'],
                    batch_size=batch_size,device=device,precision=precision)
            assert on['source']['fingerprint'] == off['source']['fingerprint']
            def compact(metrics):
                return {key:metrics[key] for key in
                        ('cross_entropy','final_accuracy','token_accuracy','sequence_exact_accuracy','source')} | {
                            'final_cross_entropy':metrics['final']['cross_entropy']}
            result['lengths'][str(length)] = dict(on=compact(on),off=compact(off),
                off_minus_on_cross_entropy=off['cross_entropy']-on['cross_entropy'],
                off_minus_on_final_cross_entropy=off['final']['cross_entropy']-on['final']['cross_entropy'],
                on_minus_off_final_accuracy=on['final_accuracy']-off['final_accuracy'],
                layers=stats.summary())
    finally:
        model.set_phase_correction_scale(scale)
        model.train(training)
        restore_rng(rng)
    return result


def flatten_diagnostics(report):
    result = {}
    def visit(prefix, value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key != 'source':
                    visit(prefix+'_'+key,item)
        elif isinstance(value,(int,float,bool)):
            result[prefix] = value
    for length,entry in report['lengths'].items():
        visit('length_'+length,entry)
    return result
