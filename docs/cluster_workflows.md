# Optional Slurm execution

The same local CLI can run inside a GPU allocation. Generate publication configs
and prepare shared statistics first, as described in `publication/README.md`.
Install the locked environment on the compute system and place Census data at the
configured path. Run from the repository root.

`run/slurm.sh` accepts one configuration and, optionally, `train` or
`train_with_downstream`. Supply partition, account, time, CPU and memory requests
appropriate for your site. For example:

```sh
sbatch --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00 \
  run/slurm.sh configs/scgpt_like/config_0.yml train_with_downstream
```

These resource requests are examples, not validated requirements for every model.
The script uses `.venv/bin/python`, or the executable selected by
`REPRO_PYTHON`. It does not install environments, transfer data, delete run outputs,
or automatically resubmit jobs. To resume, invoke the local CLI with `--resume`
inside a new allocation. Use separate configurations for concurrent jobs.
