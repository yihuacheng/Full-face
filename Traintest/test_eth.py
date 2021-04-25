import model
import importlib
import numpy as np
import cv2 
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import yaml
import os
import copy

def gazeto3d(gaze):
  gaze_gt = np.zeros([3])
  gaze_gt[0] = np.cos(gaze[0]) * np.sin(gaze[1])
  gaze_gt[1] = np.sin(gaze[0])
  gaze_gt[2] = np.cos(gaze[0]) * np.cos(gaze[1])
  return gaze_gt

def angular(gaze, label):
  total = np.sum(gaze * label)
  return np.arccos(min(total/(np.linalg.norm(gaze)* np.linalg.norm(label)), 0.9999999))*180/np.pi

if __name__ == "__main__":
  config = yaml.load(open(sys.argv[1]), Loader = yaml.FullLoader)
  readername = config["reader"]
  dataloader = importlib.import_module("reader." + readername)

  config = config["test"]
  imagepath = config["data"]["image"]
  labelpath = config["data"]["label"]
  modelname = config["load"]["model_name"] 
  
  loadpath = os.path.join(config["load"]["load_path"])
  device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

  savepath = os.path.join(loadpath, f"checkpoint")
  
  if not os.path.exists(os.path.join(loadpath, f"prediction")):
    os.makedirs(os.path.join(loadpath, f"prediction"))

  print("Read data")
  dataset = dataloader.txtload(labelpath, imagepath, 32, num_workers=4, train=False, header=True)

  begin = config["load"]["begin_step"]
  end = config["load"]["end_step"]
  step = config["load"]["steps"]

  for saveiter in range(begin, end+step, step):
    print("Model building")
    net = model.model()
    statedict = torch.load(os.path.join(savepath, f"Iter_{saveiter}_{modelname}.pt"), map_location={"cuda:0":"cuda:1"})

    net.to(device)
    net.load_state_dict(statedict)
    net.eval()

    print(f"Test {saveiter}")
    length = len(dataset)
    accs = 0
    count = 0
    with torch.no_grad():
      with open(os.path.join(loadpath, f"prediction/{saveiter}.log"), 'w') as outfile:
        for j, data in enumerate(dataset):
          img = data["face"].to(device) 
          names =  data["name"]

          img = {"face":img}
           
          gazes = net(img)
          for k, gaze in enumerate(gazes):
            gaze = gaze.cpu().detach().numpy()
            count += 1

            name = [names[k]]
            gaze = [str(u) for u in gaze] 
            log = name + [",".join(gaze)]
            outfile.write(" ".join(log) + "\n")

