import urllib3
import json
import ast
import os

data_dir = "data/harvard_art_museum"

with open("harvard_vanitas_results.json", "r") as infile:
    vanitas_list = json.load(infile)

http = urllib3.PoolManager()

for obj in vanitas_list["records"]:
    obj_id = obj["id"]
    # Find all of the objects with the word "cat" in the title and return only a few fields per record
    r = http.request(
        "GET",
        "https://api.harvardartmuseums.org/object/{}".format(str(obj_id)),
        fields={
            "apikey": "fc5d8b76-cf55-4a49-9946-315cd5786a66",
            "title": "vanitas",
            "fields": "objectnumber,title,dated,contextualtext",
        },
    )

    # print(r.status, r.data)
    object_dict = json.loads(r.data.decode("utf-8"))

    save_dir = os.path.join(data_dir, str(obj_id) + ".json")

    with open(save_dir, "w") as output_file:
        json.dump(object_dict, output_file)

    # print(object_dict["contextualtext"][0]["text"])

