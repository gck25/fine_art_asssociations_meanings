# print(response.json()["contextualtext"][0]["text"])

import urllib3
import json
import ast

http = urllib3.PoolManager()

# Find all of the objects with the word "cat" in the title and return only a few fields per record
r = http.request(
    "GET",
    "https://api.harvardartmuseums.org/object",
    fields={
        "apikey": "fc5d8b76-cf55-4a49-9946-315cd5786a66",
        "title": "vanitas",
        "fields": "objectnumber,title,dated",
    },
)

print(r.status, r.data)

with open("harvard_vanitas_results.json", "w") as outfile:
    json.dump(ast.literal_eval(r.data.decode("utf-8")), outfile)
