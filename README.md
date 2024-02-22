## The offical PyTorch code for paper ["Knowledge-Aided Momentum Contrastive Learning for Remote-Sensing Image Text Retrieval", TGRS 2023.](https://doi.org/10.1109/TGRS.2023.3332317)

# KAMCL
##### Author: Changxu Meng

![Supported Python versions](https://img.shields.io/badge/python-3.9-blue.svg)
![Supported OS](https://img.shields.io/badge/Supported%20OS-Linux-yellow.svg)
![npm License](https://img.shields.io/npm/l/mithril.svg)

### -------------------------------------------------------------------------------------

## INTRODUCTION
Remote-sensing image–text retrieval (RSITR) has attracted widespread attention due to its great potential for rapid information mining ability on remote-sensing images. Although significant progress has been achieved, existing methods typically overlook the challenge posed by the extremely analogous descriptions, where the subtle differences remain largely unexploited or, in some cases, are entirely disregarded. To address the limitation, we propose a knowledge-aided momentum contrastive learning (KAMCL) method for RSITR. Specifically, we propose a novel knowledge-aided learning (KAL) framework, including knowledge initialization, construction, filtration, and alignment operations, which aims at providing valuable concepts and learning discriminative representations. On this basis, we integrate momentum contrastive learning (MCL) to promote the capture of key concepts within the representation via expanding the scale of negative sample pairs. Moreover, we designed a hierarchical aggregator (HA) module to better capture the multilevel information from remote-sensing images. Finally, we introduce an innovative two-step training strategy designed to effectively harness the synergy among concepts and leverage their respective functionalities. Extensive experiments conducted on the three public datasets showcase the remarkable performance of our approach in terms of retrieval accuracy and computational efficiency. For instance, compared with the existing state-of-the-art method, our method exhibits notable performance improvements of 2.65% on the RSICD dataset, simultaneously achieving improvements in inference efficiency by 48%. 

![overview](./figure/overview.jpg)

### Performance
![performance](./figure/rsicd_rsitmd.jpg)
![performance](./figure/nwpu.jpg)
Comparisons of Retrieval Performance on RSICD, RSITMD and NWPU-Captions Testset.
### -------------------------------------------------------------------------------------
## Acknowledgement
We thank Zhiqiang Yuan and his GaLR framework for assisting us in building our own structure.(https://github.com/xiaoyuan1996/GaLR)

## Citation
If you feel this code helpful or use this code or dataset, please cite it as
```
Z. Ji, C. Meng, Y. Zhang, Y. Pang and X. Li, "Knowledge-Aided Momentum Contrastive Learning for Remote-Sensing Image Text Retrieval," in IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-13, 2023, Art no. 5625213, doi: 10.1109/TGRS.2023.3332317
```
