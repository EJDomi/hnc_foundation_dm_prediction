layer_loop = {}
layer_loop_downsample = {}
for j in range(3):
    layer_loop[f"layer1.{j}"] = f"{0}.{j}"
for j in range(4):
    layer_loop[f"layer2.{j}"] = f"{1}.{j}"
for j in range(3):
    layer_loop[f"layer3.{j}"] = f"{2}.{j}"
for j in range(3):
    layer_loop[f"layer4.{j}"] = f"{3}.{j}"

layer_loop_downsample['downsample.0'] = 'downsample'
layer_loop_downsample['downsample.1'] = 'bn0'


