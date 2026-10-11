"""Smoke checks for the final Strict byte model and its training/evaluation path."""
import json

import pytest
import torch
from transformers import ByT5Tokenizer

from babylm.common import ROOT, encode_text, load_tokenizer, make_tokenizer, read_json
from babylm.eval_blimp import evaluate, parser as eval_parser, score_sentences
from babylm.prepare import prepare
from babylm.train import lr_at, parser as train_parser, train
from configuration_trace_relay import TraceRelayConfig
from modeling_trace_relay import TraceRelayForCausalLM


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def final_config():
    return TraceRelayConfig.from_dict(read_json(ROOT / 'babylm/configs/unet16_102m.json'))


def test_final_model_tokenizer_and_lr(tmp_path):
    config = final_config()
    assert config.num_hidden_layers == 16 and config.hidden_sizes == [736, 368, 192, 368, 736]
    assert {tuple(pair) for pair in config.skip_stage_pairs} == {(0, 4), (1, 3)}
    assert config.left_window == [16, 32, 64, 128, 256, 512, 1024, 2048, 2048, 1024, 512, 256, 128, 64, 32, 16]
    assert config.right_window == [16, 32, 64, 128, 256, 256, 256, 256, 256, 256, 256, 256, 128, 64, 32, 16]
    assert config.relay_stride == [r + 1 for r in config.right_window]
    with torch.device('meta'):
        model = TraceRelayForCausalLM(config)
    assert sum(p.numel() for p in model.parameters()) == 102_131_392
    assert model.get_input_embeddings().weight is not model.get_output_embeddings().weight
    tokenizer = make_tokenizer()
    tokenizer.save_pretrained(tmp_path)
    loaded = load_tokenizer(tmp_path)
    text = 'Hello café 🙂 </s>'
    assert type(loaded) is ByT5Tokenizer and len(loaded) == 259
    assert encode_text(loaded, text) == [byte + 3 for byte in text.encode('utf-8')]
    assert loaded.decode(encode_text(loaded, text)) == text
    assert (loaded.pad_token_id, loaded.bos_token_id, loaded.eos_token_id) == (0, 1, 1)
    for step, expected in [(100, 1.5e-4), (16440, 1.5e-5), (32880, 1.5e-4), (49320, 1.5e-5)]:
        assert 1.5e-4 * lr_at(step, 16440, 100, .1, cycle_steps=16440) == pytest.approx(expected)


def test_prepare_train_save_and_blimp(tmp_path):
    # A small fixture exercises the actual entry points without requiring a GPU.
    source, heldout = tmp_path/'train.txt', tmp_path/'validation.txt'
    source.write_text('The cat runs. A dog sleeps.\n' * 64)
    heldout.write_text('The dog runs. A cat sleeps.\n' * 4)
    data = tmp_path/'data'
    manifest = prepare(data, [source], validation_files=[heldout])
    assert manifest['track'] == '2026-strict' and manifest['max_train_words'] == 100_000_000
    config = TraceRelayConfig(vocab_size=259, hidden_sizes=[16, 8, 16], depths=[1, 1, 1],
                             stage_num_attention_heads=[2, 1, 2], mlp_ratios=2,
                             left_windows=[2, 4, 2], right_windows=1, relay_strides=2,
                             skip_stage_pairs=[[0, 2]], pad_token_id=0, bos_token_id=1, eos_token_id=1)
    config_path = tmp_path/'config.json'
    config_path.write_text(json.dumps(config.to_dict()))
    output = tmp_path/'run'
    args = train_parser().parse_args([
        '--data', str(data), '--config', str(config_path), '--output', str(output),
        '--device', 'cpu', '--precision', 'float32', '--attn-implementation', 'eager',
        '--epochs', '3', '--stop-after-steps', '2', '--sequence-length', '64',
        '--batch-size', '2', '--effective-batch-size', '2', '--eval-batch-size', '2',
        '--learning-rate', '0.00015', '--min-lr-ratio', '0.1', '--warmup-steps', '100',
        '--lr-schedule', 'cosine_cycle', '--cosine-half-cycle-epochs', '1.001157',
        '--weight-decay', '0.1', '--grad-clip', '1.0', '--seed', '42', '--cpu-threads', '1'])
    summary = train(args)
    assert summary['step'] == 2 and torch.isfinite(torch.tensor(summary['final_validation']['loss']))
    assert (output/'best/model.safetensors').is_file() and (output/'last/training_state.pt').is_file()
    blimp = tmp_path/'blimp'
    blimp.mkdir()
    (blimp/'agreement.jsonl').write_text(json.dumps({
        'sentence_good': 'The cat runs.', 'sentence_bad': 'The cat run.'}))
    result = evaluate(eval_parser().parse_args([
        '--checkpoint', str(output/'best'), '--data', str(blimp),
        '--output', str(tmp_path/'evaluation'), '--device', 'cpu']))
    assert result['task_count'] == result['pair_count'] == 1
    assert 0 <= result['macro_accuracy'] <= 1 and result['metadata']['vocab_size'] == 259
    assert all((tmp_path/'evaluation'/name).is_file()
               for name in ['results.json', 'metrics.csv', 'predictions.jsonl'])


@pytest.mark.cuda
def test_final_setting_cuda_update_and_blimp():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required for the final model smoke test')
    pytest.importorskip('flash_attn')
    torch.manual_seed(42)
    model = TraceRelayForCausalLM(final_config()).float().cuda()
    model.set_attn_implementation('flash_attention_2')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, betas=(.9, .95), weight_decay=.1)
    # Final sequence length 2048, microbatch 11 + 5, effective batch 16.
    stream = torch.randint(3, 259, (16, 2049), device='cuda')
    for start, end in [(0, 11), (11, 16)]:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids=stream[start:end, :-1], use_cache=False).logits
        loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), stream[start:end, 1:].flatten())
        assert torch.isfinite(loss)
        (loss * ((end-start)/16)).backward()
        del logits, loss
    assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
    assert model.get_input_embeddings().weight.grad is not None
    assert model.get_output_embeddings().weight.grad is not None
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    scores = score_sentences(model, make_tokenizer(), ['The cat runs.', 'The cat run.'],
                             device='cuda', precision='bfloat16')
    assert len(scores) == 2 and torch.isfinite(torch.tensor(scores)).all()
    del optimizer, model
    torch.cuda.empty_cache()
