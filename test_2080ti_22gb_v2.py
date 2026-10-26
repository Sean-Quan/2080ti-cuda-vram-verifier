"""CUDA VRAM allocation and full read/write verification for modified 22 GiB GPU.
Run: py -3.14 test_2080ti_22gb_v2.py
Uses small CPU readback chunks, avoiding large GPU comparison intermediates.
"""
import gc
import time
import torch

GIB = 1024 ** 3
MIB = 1024 ** 2
BLOCK_MIB = 256
CHECK_MIB = 8
TARGETS_GIB = (12, 16, 18, 20)

if not torch.cuda.is_available():
    raise SystemExit('CUDA GPU not available in PyTorch.')

torch.cuda.set_device(0)
print('GPU:', torch.cuda.get_device_name(0), flush=True)
print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda, flush=True)
free, total = torch.cuda.mem_get_info()
print(f'Total VRAM: {total/GIB:.2f} GiB | Initially free: {free/GIB:.2f} GiB', flush=True)
print('Testing GPU allocation + writing every byte + readback/verification on CPU.')
print('Note: OOM means the allocation cannot finish in current Windows environment; it does NOT prove faulty VRAM.\n', flush=True)

for target in TARGETS_GIB:
    blocks = []
    status = 'FAIL'
    start = time.perf_counter()
    try:
        free, total = torch.cuda.mem_get_info()
        print(f'=== {target} GiB test | CUDA free before: {free/GIB:.2f} GiB ===', flush=True)
        if free < target * GIB + 128 * MIB:
            print('WARNING: Limited free VRAM; test may OOM.', flush=True)
        for i in range(target * 1024 // BLOCK_MIB):
            t = torch.empty(BLOCK_MIB * MIB, dtype=torch.uint8, device='cuda')
            t.fill_((i * 37 + 19) % 256)
            blocks.append(t)
        torch.cuda.synchronize()
        print(f'Allocated and filled {len(blocks)*BLOCK_MIB/1024:.2f} GiB.', flush=True)
        for i, t in enumerate(blocks):
            expected = (i * 37 + 19) % 256
            for offset in range(0, BLOCK_MIB * MIB, CHECK_MIB * MIB):
                part = t.narrow(0, offset, CHECK_MIB * MIB).cpu()
                if not bool(torch.all(part == expected).item()):
                    raise ValueError(f'MISMATCH at block {i}, offset {offset // MIB} MiB')
                del part
            if (i + 1) % 8 == 0:
                print(f'  Verified {(i+1)*BLOCK_MIB/1024:.0f}/{target} GiB', flush=True)
        torch.cuda.synchronize()
        status = 'PASS'
        print(f'PASS: {target} GiB allocated, written, and ALL bytes read back and verified. ({time.perf_counter()-start:.1f}s)\n', flush=True)
    except torch.cuda.OutOfMemoryError as exc:
        print(f'OOM: {target} GiB could not finish (GPU memory budget/overhead).', flush=True)
        print('This is NOT a memory corruption result.\n', flush=True)
    except Exception as exc:
        print(f'ERROR: {target} GiB test: {type(exc).__name__}: {exc}\n', flush=True)
    finally:
        del blocks
        gc.collect()
        torch.cuda.empty_cache()

print('Done. PASS verifies CUDA memory allocation/read/write for that target; it is not a long-term burn-in guarantee.', flush=True)
input('Press Enter to close...')
