import logging
import argparse

# Set up file handler explicitly since basicConfig is a no-op if root logger already has handlers
logging.basicConfig(level=logging.INFO)
file_handler = logging.FileHandler('run_all_bitfiles.log', mode='a')
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
))
logging.root.addHandler(file_handler)
_LOGGER = logging.getLogger(__name__)

from tmu.data import MNIST
from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
from tmu.tools import BenchmarkTimer

def metrics(args):
    return dict(
        number_of_positive_clauses=[],
        accuracy=[],
        number_of_includes=[],
        train_time=[],
        test_time=[],
        args=vars(args)
    )

def main(args):
    experiment_results = metrics(args)

    data = MNIST().get()
    
    # Reduce dataset size to a tenth
    train_size = len(data["x_train"])
    test_size = len(data["x_test"])
    data["x_train"] = data["x_train"][:train_size]
    data["y_train"] = data["y_train"][:train_size]
    data["x_test"] = data["x_test"][:test_size]
    data["y_test"] = data["y_test"][:test_size]

    tm = TMCoalescedClassifier(
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        platform=args.platform,
        weighted_clauses=args.weighted_clauses,
        focused_negative_sampling=args.focused_negative_sampling,
        patch_dim=args.dim,
        seed=1,
        bitfile_path=args.bitfile_path
    )

    _LOGGER.info(f"Running {TMCoalescedClassifier} for {args.epochs} epochs with bitfile {args.bitfile_path}")
    for epoch in range(args.epochs):

        benchmark1 = BenchmarkTimer()
        with benchmark1:
            tm.fit(data["x_train"], data["y_train"])
        experiment_results["train_time"].append(benchmark1.elapsed())

        benchmark2 = BenchmarkTimer()
        with benchmark2:
            result = 100 * (tm.predict(data["x_test"]) == data["y_test"]).mean()
            experiment_results["accuracy"].append(result)
        experiment_results["test_time"].append(benchmark2.elapsed())

        number_of_positive_clauses = 0
        for i in range(tm.number_of_classes):
            number_of_positive_clauses += (tm.weight_banks[i].get_weights() > 0).sum()
        number_of_positive_clauses /= tm.number_of_classes
        experiment_results["number_of_positive_clauses"].append(number_of_positive_clauses)

        number_of_includes = 0
        for j in range(args.num_clauses):
            number_of_includes += tm.number_of_include_actions(j)
        number_of_includes /= 2 * args.num_clauses
        experiment_results["number_of_includes"].append(number_of_includes)

        _LOGGER.info(
            f"Epoch: {epoch + 1}, "
            f"Accuracy: {result:.2f}, "
            f"Positive clauses: {number_of_positive_clauses}, "
            f"Literals: {number_of_includes}, "
            f"Training Time: {benchmark1.elapsed():.2f}s, "
            f"Testing Time: {benchmark2.elapsed():.2f}s"
        )

    return experiment_results


# def default_args(**kwargs):
#     default_clauses = 64
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--num-clauses", default=default_clauses, type=int)
#     parser.add_argument("--T", default=default_clauses // 4, type=int)
#     parser.add_argument("--s", default=10.0, type=float)
#     parser.add_argument("--weighted-clauses", default=True, type=bool)
#     parser.add_argument("--platform", default='FPGA', type=str)
#     parser.add_argument("--focused-negative-sampling", default=True, type=bool)
#     parser.add_argument("--epochs", default=60, type=int)
#     parser.add_argument("--dim", default=(10, 10), type=tuple)
#     parser.add_argument("--bitfile-path", default=None, type=str)
#     args = parser.parse_args()
#     for key, value in kwargs.items():
#         if key in args.__dict__:
#             setattr(args, key, value)
#     return args


if __name__ == "__main__":
    # Iterate through different bitfiles
    for patch_dim in [(5, 5), (10, 10), (5, 10), (15, 15)]:
        for clause_count in [64, 128, 256, 512, 1024, 2048]:
            for w_platform in ['FPGA', 'CPU']:
                _LOGGER.info(f"Running bitfile for patch_dim={patch_dim} and clause_count={clause_count} on platform={w_platform}")
                bitfile_path = f"/home/xilinx/modded_tmu/bitfiles/{patch_dim[0]:02d}x{patch_dim[1]:02d}_{clause_count:04d}_14_1/TM_Inference.bit"
                args = argparse.Namespace(
                    num_clauses=clause_count,
                    T=clause_count // 4,
                    s=10.0,
                    weighted_clauses=True,
                    platform=w_platform,
                    focused_negative_sampling=True,
                    epochs=10,
                    dim=patch_dim,
                    bitfile_path=bitfile_path
                )
                results = main(args)
                _LOGGER.info(results)
