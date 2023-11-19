import wget
import os


TASK_A_CSV2URL = {
    'homo_eng_3_train.csv': 'https://drive.google.com/u/0/uc?id=1-4tEQPEnorw76RbHTtQHCETbCFyvTNOV&export=download',
    'homo_eng_3_dev.csv': 'https://drive.google.com/u/0/uc?id=1-49xQzAz4kWR7PL4hwDOwQ_rJPL8WqT1&export=download',
    'homo_eng_3_test.csv': 'https://drive.google.com/u/0/uc?id=1--OKJ81BP2Yo-bDgiJaOd7HKyx58rBac&export=download',
    'homo_tam_3_train.csv': 'https://drive.google.com/u/0/uc?id=1-qmbtuVhbTK8CfUwAhSJDEIuukkM1JHS&export=download',
    'homo_tam_3_dev.csv': 'https://drive.google.com/u/0/uc?id=1-ge3p3LRIrlgNjW06ZCb77uRYymUYC-9&export=download',
    'homo_tam_3_test.csv': 'https://drive.google.com/u/0/uc?id=1-eycyrq3-3C93aKwkd3wkDHldq83k9bi&export=download',
    'homo_hin_3_train.csv': 'https://drive.google.com/u/0/uc?id=1i1eB3T0ECBnT91G1bIuuAGwNiFyTfMt8&export=download',
    'homo_hin_3_dev.csv': 'https://drive.google.com/u/0/uc?id=1sqCdTf3hf84OtSaldoCfjQ_zWlckOsOl&export=download',
    'homo_hin_3_test.csv': 'https://drive.google.com/u/0/uc?id=1-ENFYB0NjXS7YMO04Im_riht2vP1vTlH&export=download',
    'homo_spanish_3_train.csv': 'https://drive.google.com/u/0/uc?id=1iFEh0w_73OTjq8_YI1u9kgTFDKK19Utx&export=download',
    'homo_spanish_3_dev.csv': 'https://drive.google.com/u/0/uc?id=1V-B1zRukn6wVAwCvuQ5sTaqSfqwad7Sw&export=download',
    'homo_spanish_3_test.csv': 'https://drive.google.com/u/0/uc?id=18d8c6Ks7XJ26jPJI3TzraaEq9uugIHcU&export=download',
    'homo_mal_3_train.csv': 'https://drive.google.com/u/0/uc?id=1-azHdEC2px1QsEvIu5wyCewVHiM29r4L&export=download',
    'homo_mal_3_dev.csv': 'https://drive.google.com/u/0/uc?id=1-_xQ5NoJS1p4pFXFHUUGKGGzELIV-AJ0&export=download',
    'homo_mal_3_test.csv': 'https://drive.google.com/u/0/uc?id=1-XBo4kwcBE7GWlDFjUqK8P9E0xKbxs5m&export=download',
}


def download_task_a(dir_, lang="", split=""):
    for out, url in TASK_A_CSV2URL.items():
        _, l, _, s = out.split("_")[:4]
        if (lang in l) or (split in s):
            path = os.path.join(dir_, out)
            if not os.path.isfile(path):
                wget.download(url, out=path)


if __name__ == "__main__":
    download_task_a('./data')
