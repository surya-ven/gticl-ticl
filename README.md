# GTICL and TICL Implementation
GTICL was an attempt to extend TICL by also using a tuning-free method but this time mirroring the GRPO algorithm for sampling and ranking negative examples.

Please see the TICL paper below for more details on the basis of the extension and the GTICL source code for the extension.

# Training and Evaluation
Training and evaluation code can be found in the `compare_gticl_ticl.ipynb` notebook. The `results` folder contains 2 runs of the tests in the notebook.  

# TICL
This was my implementation of the TICL algorithm as shown in:

@misc{cho2025tuningfreepersonalizedalignmenttrialerrorexplain,
      title={Tuning-Free Personalized Alignment via Trial-Error-Explain In-Context Learning}, 
      author={Hyundong Cho and Karishma Sharma and Nicolaas Jedema and Leonardo F. R. Ribeiro and Alessandro Moschitti and Ravi Krishnan and Jonathan May},
      year={2025},
      eprint={2502.08972},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.08972}, 
}
