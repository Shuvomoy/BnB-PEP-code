# BnB-PEP

This code can be used to reproduce and verify the results from the paper

> [Shuvomoy Das Gupta, Bart P.G. Van Parys, Ernest K. Ryu, "Branch-and-Bound Performance Estimation Programming: A Unified Methodology for Constructing Optimal Optimization Methods", Mathematical Programming 204.1 (2024): 567-639.](https://link.springer.com/article/10.1007/s10107-023-01973-1)

A preprint of the work is available on arXiv [here](https://arxiv.org/abs/2203.07305). 

If you find the code helpful in you project, we kindly request that you cite the following paper:

```
@article{dasgupta2022BnBPEP,
  title={Branch-and-bound performance estimation programming: A unified methodology for constructing optimal optimization methods},
  author={Das Gupta, Shuvomoy and Van Parys, Bart PG and Ryu, Ernest K},
  journal={Mathematical Programming},
  volume={204},
  number={1},
  pages={567--639},
  year={2024},
  publisher={Springer}
}
```

**Please use Gurobi 12.0 or later.** Please use `Gurobi 12.0` or later when implementing stage 3 of the BnB-PEP algorithm: several issues in the spatial branch-and-bound algorithm in `Gurobi 10.0` that cause numerical instabilities have recently been fixed; see the list of bug fixes [here](https://docs.gurobi.com/projects/optimizer/en/12.0/reference/releasenotes/fixedbugs.html) and [here](https://docs.gurobi.com/projects/optimizer/en/11.0/reference/releasenotes/fixedbugs.html).

## Reporting issues

Please report any issues via the [Github issue tracker](https://github.com/Shuvomoy/BnB-PEP-code/issues). All types of issues are welcome including bug reports, feature requests, and so on.

## Contact
Please feel free to send an email :email: to [sd158@rice.edu](sd158@rice.edu) regarding any subject including but not limited to comments about this paper, performance estimation problems in general, implementation for a specific research problem, or just to say hi 😃!