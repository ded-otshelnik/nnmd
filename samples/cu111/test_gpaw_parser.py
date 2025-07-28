from nnmd.io import input_parser

text_file = "input/input.yaml"

data = input_parser(text_file)
print(len(data["atomic_data"]["reference_data"]))
for data in data["atomic_data"]["reference_data"]:
    print(data.keys())
    assert len(data.keys()) > 2, "Data should contain forces and energy"
    print(data["Cu"]["positions"])
