<h1 align="center"> Linguistic Laws Meet Protein Sequences: A Comparative Analysis of Subword Tokenization Methods </h1>
<h4 align="center"> The implementation of the paper "Linguistic Laws Meet Protein Sequences: A Comparative Analysis of Subword Tokenization Methods" </h4>


## Overview 

Tokenization is a crucial step in processing protein sequences for machine learning models, as proteins are complex sequences of amino acids that require meaningful segmentation to capture their functional and structural properties. However, existing subword tokenization methods, developed primarily for human language, may be inadequate for protein sequences, which have unique patterns and constraints.
This paper investigates the effectiveness of three popular tokenization methods—Byte-Pair Encoding (BPE), WordPiece, and SentencePiece—when applied to protein sequences from the UniRef50 dataset. We evaluate these tokenizers across various vocabulary sizes (400–6400), analyzing their ability to align with protein domain boundaries, handle vocabulary scaling, and adhere to linguistic laws such as Zipf’s, Brevity, Heaps’, and Menzerath’s laws, which may offer insights into the structural organization of proteins.
Our experiments show that BPE maintains protein domain boundaries better and has greater contextual diversity, while SentencePiece provides more consistent token lengths and lower fertility scores. Despite these advantages, all tokenization methods face challenges in fully preserving protein domain integrity as vocabulary size increases. Additionally, while these tokenizers demonstrate general adherence to linguistic laws, the variability in their behavior highlights the unique demands of protein sequences compared to natural language. These findings underscore the need for domain-specific optimizations to enhance tokenization strategies for protein sequence modeling, contributing to the broader effort to apply natural language processing techniques to bioinformatics.

## Reference

If you use this repository, please cite the following related [paper]():
```bibtex
@article{,
  title={},
  author={},
  journal={},
  year={},
  publisher={}
}
```


## License

This code base is licensed under the MIT license. See [LICENSE](license.md) for details.