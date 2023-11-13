# LanguagePaint
***In Development***

This repository contains the implementation code for paper: <br>
**Patching Language-Specific Homophobia/Transphobia Classifiers with a Multilingual Understanding**
Dean Ninalga <br>
*Proceedings of the 14th International Conference on Recent Advances in Natural Language Processing, 2023* <br>
[[Website](https://aclanthology.org/events/ranlp-2023/)] [[Paper](https://aclanthology.org/2023.ltedi-1.28.pdf)]

`1st place` in three of the five languages in **Task A** in the **Second Shared Task on Homophobia and Transphobia Detection in Social Media Comments** <br>
[[Overview Paper](https://aclanthology.org/2023.ltedi-1.6.pdf)] [[Dataset Paper](https://www.sciencedirect.com/science/article/pii/S2667096822000623)] [[Website](https://codalab.lisn.upsaclay.fr/competitions/11077#learn_the_details-overview)]

If you find this idea useful, please consider citing:
```bib
@inproceedings{ninalga-2023-cordyceps,
    title = "Cordyceps@{LT}-{EDI}: Patching Language-Specific Homophobia/Transphobia Classifiers with a Multilingual Understanding",
    author = "Ninalga, Dean",
    editor = "Chakravarthi, Bharathi R.  and
      Bharathi, B.  and
      Griffith, Joephine  and
      Bali, Kalika  and
      Buitelaar, Paul",
    booktitle = "Proceedings of the Third Workshop on Language Technology for Equality, Diversity and Inclusion",
    month = sep,
    year = "2023",
    address = "Varna, Bulgaria",
    publisher = "INCOMA Ltd., Shoumen, Bulgaria",
    url = "https://aclanthology.org/2023.ltedi-1.28",
    pages = "185--191",
    abstract = "Detecting transphobia, homophobia, and various other forms of hate speech is difficult. Signals can vary depending on factors such as language, culture, geographical region, and the particular online platform. Here, we present a joint multilingual (M-L) and language-specific (L-S) approach to homophobia and transphobic hate speech detection (HSD). M-L models are needed to catch words, phrases, and concepts that are less common or missing in a particular language and subsequently overlooked by L-S models. Nonetheless, L-S models are better situated to understand the cultural and linguistic context of the users who typically write in a particular language. Here we construct a simple and successful way to merge the M-L and L-S approaches through simple weight interpolation in such a way that is interpretable and data-driven. We demonstrate our system on task A of the {``}Shared Task on Homophobia/Transphobia Detection in social media comments{''} dataset for homophobia and transphobic HSD. Our system achieves the best results in three of five languages and achieves a 0.997 macro average F1-score on Malayalam texts.",
}
```
