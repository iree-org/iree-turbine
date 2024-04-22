# Running Tests

## Set Up
This test suite requires a local clone of the `pytorch-jit-paritybench` repository:
```shell
# somewhere ...
git clone https://github.com/jansel/pytorch-jit-paritybench.git
cd pytorch-jit-paritybench
pip install -r requirements.txt
conda install cpuonly -c pytorch-nightly
```

Note that we are not exactly following the setup described in the above repo, mainly to avoid issues with dependencies between `conda` and `pip` versions of relevant packages (see 'Known Issues' below).

There may be some additional packages to install that are not in the requirements.txt in order to successfully run the tests, one that comes to mind is `expecttest`

## Running
Once everything is set up in your conda environment we can run the test suite using `python/test/generated/main.py`. Initially you have to pass the location of your `pytorch-jit-paritybench` repo to the script: `python main.py --tests-dir /path/to/pytorch-jit-paritybench`. After the first run, the script will save the path for future use in a local text file for convenience and you do not need to pass it again.

To speed up iteration on tests it's recommended to make use of the `offset`, `limit`, and `filter` arguments as running the full test suite can take some time.

## Unimplemented Torch Ops
Many of the errors in our test suite will arise due to unimplemented ops in torch-mlir, we can use the `extract_unimpl_ops.sh` script to extract a list of these ops:
```bash
python main.py --limit 500 -j 8 | grep "NotImplementedError: Unimplemented torch op in the IREE compiler" | grep -o "'[^']*'" | sed "s/'//g" > unimplemented_torch_ops.txt
```


## Help
```
usage: main.py [-h] [--jobs JOBS] [--offset OFFSET] [--limit LIMIT] [--filter FILTER] [--skips SKIPS] [--tests-dir TESTS_DIR]

options:
  -h, --help            show this help message and exit
  --jobs JOBS, -j JOBS  Number of threads in our threadpool, jobs=1 is essentially sequential execution
  --offset OFFSET       Pick files starting from this offset. Together with --limit, we can run through all files in multiple separate runs
  --limit LIMIT, -l LIMIT
                        only run the first N files
  --filter FILTER, -f FILTER, -k FILTER
                        only run module containing given name
  --skips SKIPS
  --tests-dir TESTS_DIR
                        jit-paritybench location (i.e. /path/to/pytorch-jit-paritybench)

```


# Known Issues
On Mac, setting resource limits via a python shell is finicky/not allowed this can cause issues with the jit parity-bench tests as they utilize the `resource` package to set an optional resource limit [here](https://github.com/jansel/pytorch-jit-paritybench/blob/7e55a422588c1d1e00f35a3d3a3ff896cce59e18/paritybench/utils.py#L57) - the simplest fix is to simply comment those lines out in your local `jit-paritybench` repo.

Getting unknown symbol errors associated with a shared library (commonly a torch library) often occurs because of mixing conda installed dependencies with pip installed dependencies because of possible differences in how certain shared libraries are linked, if possible use pip for all of your dependencies (especially when they have a dependence on one another like torch, torchvision, and torchaudio)