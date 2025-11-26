# run_all_evals.py
"""
Unified script to run all evaluations.
Automatically detects all checkpoints and runs appropriate evaluation scripts.
"""

import os
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_checkpoints(models_dir="../models"):
    """
    Find all checkpoint files in the models directory.

    Returns:
        dict: Categorized checkpoints
    """
    checkpoints = {
        'baseline': [],
        'multitask': []
    }

    models_path = Path(models_dir)

    if not models_path.exists():
        logger.warning(f"Models directory not found: {models_dir}")
        return checkpoints

    # Find all .pt files
    for pt_file in models_path.rglob("*.pt"):
        file_name = pt_file.name.lower()

        if 'baseline' in file_name:
            checkpoints['baseline'].append(str(pt_file))
        elif 'multitask' in file_name or 'multi' in file_name:
            checkpoints['multitask'].append(str(pt_file))

    return checkpoints


def run_baseline_eval(checkpoint_path, output_csv="../results/baseline_eval_results.csv", batch_size=32):
    """Run baseline evaluation."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating Baseline: {checkpoint_path}")
    logger.info(f"{'='*60}")

    cmd = [
        "python", "eval_baseline.py",
        "--checkpoint", checkpoint_path,
        "--output", output_csv,
        "--batch_size", str(batch_size)
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        logger.info(f"✅ Baseline evaluation completed: {checkpoint_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Baseline evaluation failed: {checkpoint_path}")
        logger.error(f"Error: {e}")
        return False


def run_multitask_eval(checkpoint_path, output_csv="../results/multitask_eval_results.csv",
                       batch_size=32, eval_emotion=True):
    """Run multi-task evaluation."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating Multi-Task: {checkpoint_path}")
    logger.info(f"{'='*60}")

    cmd = [
        "python", "eval_multitask.py",
        "--checkpoint", checkpoint_path,
        "--output", output_csv,
        "--batch_size", str(batch_size)
    ]

    if eval_emotion:
        cmd.append("--eval_emotion")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        logger.info(f"✅ Multi-task evaluation completed: {checkpoint_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Multi-task evaluation failed: {checkpoint_path}")
        logger.error(f"Error: {e}")
        return False


def main():
    """Main function to run all evaluations."""

    logger.info("="*60)
    logger.info("RUNNING ALL EVALUATIONS")
    logger.info("="*60)

    # Find checkpoints
    logger.info("\nSearching for checkpoints...")
    checkpoints = find_checkpoints()

    logger.info(f"\nFound {len(checkpoints['baseline'])} baseline checkpoint(s)")
    for ckpt in checkpoints['baseline']:
        logger.info(f"  - {ckpt}")

    logger.info(f"\nFound {len(checkpoints['multitask'])} multi-task checkpoint(s)")
    for ckpt in checkpoints['multitask']:
        logger.info(f"  - {ckpt}")

    if not checkpoints['baseline'] and not checkpoints['multitask']:
        logger.warning("No checkpoints found! Make sure models are in ../models/ directory")
        return

    # Run evaluations
    results = {'baseline': [], 'multitask': []}

    # Evaluate baselines
    if checkpoints['baseline']:
        logger.info("\n" + "="*60)
        logger.info("EVALUATING BASELINE MODELS")
        logger.info("="*60)

        for ckpt in checkpoints['baseline']:
            success = run_baseline_eval(ckpt)
            results['baseline'].append((ckpt, success))

    # Evaluate multi-tasks
    if checkpoints['multitask']:
        logger.info("\n" + "="*60)
        logger.info("EVALUATING MULTI-TASK MODELS")
        logger.info("="*60)

        for ckpt in checkpoints['multitask']:
            success = run_multitask_eval(ckpt, eval_emotion=True)
            results['multitask'].append((ckpt, success))

    # Summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)

    total_success = 0
    total_attempted = 0

    if results['baseline']:
        logger.info("\nBaseline Models:")
        for ckpt, success in results['baseline']:
            status = "✅ SUCCESS" if success else "❌ FAILED"
            logger.info(f"  {status}: {os.path.basename(ckpt)}")
            total_attempted += 1
            total_success += int(success)

    if results['multitask']:
        logger.info("\nMulti-Task Models:")
        for ckpt, success in results['multitask']:
            status = "✅ SUCCESS" if success else "❌ FAILED"
            logger.info(f"  {status}: {os.path.basename(ckpt)}")
            total_attempted += 1
            total_success += int(success)

    logger.info(f"\nTotal: {total_success}/{total_attempted} evaluations successful")
    logger.info("\nResults saved to:")
    logger.info("  - ../results/baseline_eval_results.csv")
    logger.info("  - ../results/multitask_eval_results.csv")
    logger.info("="*60)


if __name__ == "__main__":
    main()
