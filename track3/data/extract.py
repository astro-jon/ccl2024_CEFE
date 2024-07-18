import json
import pandas as pd


if __name__ == '__main__':
    score2dict = {
        "不及格": 0, "一般": 1, "优秀": 2
    }
    reader = json.load(open("../../2024ccl_datas/track3/track3_val.json", "r", encoding = "utf-8"))
    idlist, titlelist, textlist, essay_scorelist = [], [], [], []
    for article in reader:
        try:idlist.append(article["id"])
        except: idlist.append(article["essay_id"])
        titlelist.append(article["title"])
        try: textlist.append(article["sent"])
        except: textlist.append(article["text"])
        try: essay_scorelist.append(score2dict[article["essay_score_level"]])
        except: essay_scorelist.append("")
    infodf = pd.DataFrame({
        "id": idlist, "title": titlelist, "text": textlist, "label": essay_scorelist
    })
    infodf.to_csv("val.csv", index = False, sep = ',')