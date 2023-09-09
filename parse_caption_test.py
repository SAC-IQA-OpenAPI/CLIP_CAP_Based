import torch
import clip
from PIL import Image
import pickle
import skimage.io as io
import pandas as pd
from tqdm.auto import tqdm
import argparse
from train_custom import ClipCaptionModel


def main(mode:str, model_path: str):
    prefix_length = 10
    clip_model_type = 'ViT-B/32'
    device = torch.device('cuda:0')
    data_path= f'../data/{mode}.csv'
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"../data/{mode}_data.pkl" # custom_path
    
    base_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    trained_model = ClipCaptionModel(prefix_length)
    trained_model.load_state_dict(torch.load(model_path, map_location=device))
    trained_model.to(device)
    data = pd.read_csv(data_path)
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(data))):
        '''
            Index(['img_name', 'img_path', dtype='object')
            img_name                                           j00zs3u6dr
            img_path                                ./test/j00zs3u6dr.jpg
            Name: 0, dtype: object
        '''
        d = data.iloc[i] # Index(['img_name', 'img_path', 'mos', 'comments'], dtype='object')
        img_id = d['img_name']
        file_name = f"../data/{mode}/{img_id}.jpg"
        image = io.imread(file_name)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = base_model.encode_image(image).to(device, dtype=torch.float32)
        all_embeddings.append(prefix)
        if mode == "train":
            all_captions.append({'caption' : d['comments'], 'image_name' : img_id, 'clip_embedding' : i})
        else:
            prefix_embed = trained_model.clip_project(prefix).reshape(1, prefix_length, -1) # shape : (1, 10, 768)
            all_captions.append({'image_name' : img_id, 'prefix_embed' : prefix_embed ,'clip_embedding' : i})
    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding" : torch.cat(all_embeddings, dim=0), "captions" : all_captions}, f)
    print('Done')
    print(f"{len(all_embeddings)} embeddings saved")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train')
    parser.add_argument('--model_path', default='./custom_coco_train/custom_prefix-031.pt')
    args = parser.parse_args()
    main(mode=args.mode, model_path=args.model_path)