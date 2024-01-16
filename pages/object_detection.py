import torch
import os
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import altair as alt

from PIL import Image
from transformers import YolosFeatureExtractor, YolosForObjectDetection
from torchvision.transforms import ToTensor, ToPILImage

st.set_page_config(layout="wide")


## CODE TO CLEAN IMAGES
def fix_channels(t):
    if len(t.shape) == 2:
        return ToPILImage()(torch.stack([t for i in (0, 0, 0)]))
    if t.shape[0] == 4:
        return ToPILImage()(t[:3])
    if t.shape[0] == 1:
        return ToPILImage()(torch.stack([t[0] for i in (0, 0, 0)]))
    return ToPILImage()(t)

## CODE FOR PLOTS WITH BOUNDING BOXES
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
        [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def idx_to_text(i):
    if i in list(dict_cats_final.keys()):
        return dict_cats_final[i.item()]
    else:
        return False

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    fig = plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        cl = p.argmax()
        if idx_to_text(cl) is False:
            pass
        else:
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
            ax.text(xmin, ymin, idx_to_text(cl), fontsize=10,
                    bbox=dict(facecolor=c, alpha=0.8))
    plt.axis('off')
    
    plt.savefig("results_od.png", 
            bbox_inches ="tight") 
    plt.show()

    st.image("results_od.png")
    
    # print matplotlib plot to streamlit
    #st.pyplot(fig)

def return_probas(outputs, threshold):
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    probas = probas[:][:,list(dict_cats_final.keys())]
    keep = probas.max(-1).values > threshold

    return probas, keep


# def visualize_predictions(image, outputs, threshold):
# # keep only predictions with confidence >= threshold
#     # convert predicted boxes from [0; 1] to image scales
#     bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)

#     # plot results
#     plot_results(image, probas[keep], bboxes_scaled)

#     return probas[keep]


def visualize_probas(probas, threshold, colors):
    label_df = pd.DataFrame({"label":probas.max(-1).indices.detach().numpy(),
                             "proba":probas.max(-1).values.detach().numpy()})
    
    cats_dict = dict(zip(np.arange(0,len(cats)),cats))
    label_df["label"] = label_df["label"].map(cats_dict)
    top_label_df = label_df.loc[label_df["proba"]>threshold].groupby("label").mean().sort_values(by=["proba"], ascending=False).round(2).reset_index()

    chart = alt.Chart(top_label_df).mark_bar().encode(x="proba", y="label", color=colors).interactive()
    st.altair_chart(chart)





######################################################################################################################################

st.markdown("# Object Detection")

st.markdown("### What is Object Detection ?")
           
#st.markdown("""Object detection involves **identifying** and **locating objects** within an image or video frame through bounding boxes. """)
st.markdown("""Object Detection is a computer vision task in which the goal is to **detect** and **locate objects** of interest in an image or video. 
            The task involves identifying the position and boundaries of objects (or **bounding boxes**) in an image, and classifying the objects into different categories.""")


st.markdown("Here is an example of Object Detection for Traffic Analysis.")
#image_od = Image.open('images/od_2.png')
#st.image(image_od, width=600)
st.video(data='https://www.youtube.com/watch?v=PVCGDoTZHaI')

st.markdown(" ")

st.markdown("""Common applications of Object Detection include:
- **Autonomous Vehicles** :car: : Object detection is crucial for self-driving cars to track pedestrians, cyclists, other vehicles, and obstacles on the road.
- **Retail** 🏬 : Implementing smart shelves and checkout systems that use object detection to track inventory and monitor stock levels.
- **Healthcare** 👨‍⚕️: Detecting and tracking anomalies in medical images, such as tumors or abnormalities, for diagnostic purposes or prevention.
- **Manufacturing** 🏭: Quality control on production lines by detecting defects or irregularities in manufactured products. Ensuring workplace safety by monitoring the movement of workers and equipment.
- **Fashion and E-commerce** 🛍️ : Improving virtual try-on experiences by accurately detecting and placing virtual clothing items on users.
""")


st.markdown("  ")
st.markdown("  ")

st.markdown("### Fashion use case")
st.markdown("""For this use case, an Object detection model was built to detect clothing in images. 
            The user is able to choose which image he wants to use for the detection, what types of clothings should be detected and what threshold to set for the model.<br> 
 """, unsafe_allow_html=True)

st.image("images/od_fashion.jpg", width=700)

#images_dior = [os.path.join("data/dior_show",url) for url in os.listdir("data/dior_show") if url != "results"]
#st.image(images_dior, width=250, caption=[file for file in os.listdir("data/dior_show") if file != "results"])

st.markdown("  ")
#st.markdown("##### Select an image")


############## SELECT AN IMAGE ###############

st.markdown("#### Step 1: Select an image")
st.markdown("""First, select the image you want to apply the object detection model to. 
            The model was trained to detect clothing items on a single person. If your image has more than individuals, the model will ignore one of them in its detection.""")

image_ = None
select_image_box = st.radio(
    "",
    ["Choose an existing image", "Load your own image"],
    index=None, label_visibility="collapsed")

if select_image_box == "Choose an existing image":
    fashion_images_path = r"data/pinterest"
    list_images = os.listdir(fashion_images_path)
    image_ = st.selectbox("", list_images, label_visibility="collapsed")
    
    if image_ is not None:
        image_ = os.path.join(fashion_images_path,image_)
        st.markdown("You've selected the following image:")
        st.image(image_, width=300)

elif select_image_box == "Load your own image":
    image_ = st.file_uploader("Load an image here", 
                                key="OD_dior", type=['jpg','jpeg','png'], label_visibility="collapsed")
    
    st.warning("""**Note**: The model tends to perform better with images of people facing forward. 
           Choose this type of image if you optimal results.""")

    if image_ is not None:
        st.image(Image.open(image_), width=300)


st.markdown("  ")
st.markdown("  ")



########## SELECT AN ELEMENT TO DETECT ##################

cats = ['shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit',
    'cape', 'glasses', 'hat', 'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar',
    'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel']

dict_cats = dict(zip(np.arange(len(cats)), cats))

st.markdown("#### Step 2: Choose the elements you want to detect")

# Select one or more elements to detect
container = st.container()
selected_options = None
all = st.checkbox("Select all")

if all:
    selected_options = container.multiselect("**Select one or more items**", cats, cats)
else:
    selected_options = container.multiselect("**Select one or more items**", cats)

#cats = selected_options 
dict_cats_final = {key:value for (key,value) in dict_cats.items() if value in selected_options}


st.markdown("  ")
st.markdown("  ")



############## SELECT A THRESHOLD ###############

st.markdown("#### Step 3: Select a threshold")

threshold = st.slider('**Select a threshold**', 0.0, 1.0, 0.05, label_visibility="collapsed")
st.warning("""**Note**: The threshold helps you decide how confident you want your model to be with its predictions. 
        Elements that were identified with a lower probability than the given threshold will be ignored in the final results. """)

st.write("You've selected a threshold at", threshold)


st.markdown("  ")


############# RUN MODEL ################

run_model = st.button("**Run the model**", type="primary")

if run_model:
    if image_ != None and selected_options != None and threshold!= None:
        with st.spinner('Wait for it...'):
            ## SELECT IMAGE
            #folder_path = r"data/dior_show"
            image = Image.open(image_)
            image = fix_channels(ToTensor()(image))

            ## LOAD OBJECT DETECTION MODEL
            MODEL_NAME = "valentinafeve/yolos-fashionpedia"
            feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small')
            model = YolosForObjectDetection.from_pretrained(MODEL_NAME)

            # RUN MODEL ON IMAGE
            inputs = feature_extractor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            probas, keep = return_probas(outputs, threshold)

            # PLOT BOUNDING BOX AND BARS/PROBA
            col1, col2 = st.columns(2)
            with col1:
                bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)
                plot_results(image, probas[keep], bboxes_scaled)
            
            with col2: 
                visualize_probas(probas, threshold, COLORS)
            
            st.info("Done")

    else:
        st.warning("You must select an **image**, **elements to detect** and a **threshold** to run the model !")


    