import streamlit as st
import os
from PIL import Image
from pixellib.tune_bg import alter_bg
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from IPython.display import Image as img
from pylab import rcParams



st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #3498DB;">
  <a class="navbar-brand" href="https://youtube.com/dataprofessor" target="_blank">Data Professor</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item active">
        <a class="nav-link disabled" href="#">Home <span class="sr-only">(current)</span></a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://youtube.com/dataprofessor" target="_blank">YouTube</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://twitter.com/thedataprof" target="_blank">Twitter</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)

st.markdown('''# **Binance Price App**
A simple cryptocurrency price app pulling price data from *Binance API*.
''')




count=0

def load_image(image_file):
	img = Image.open(image_file)
	return img


st.title("Image Upload Tutorial")
choice=None
menu = ["Image","Dataset","DocumentFiles","About"]
choice = st.selectbox("Menu",menu)

if choice == "Image":
		st.subheader("Image")
		image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

		if image_file is not None:

			  # To See details
			  file_details = {"filename":image_file.name, "filetype":image_file.type,
                              "filesize":image_file.size}
			  st.write(file_details)

              # To View Uploaded Image
			  st.image(load_image(image_file),width=500)

			  with open(os.path.join("./",image_file.name),"wb") as f:
			  	f.write((image_file).getbuffer())
			  count=1
			  st.success("File Saved")

#main model
while count==1:
	rcParams['figure.figsize'] = 10, 10
	change_bg = alter_bg()
	dir(change_bg)
	change_bg.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")


	st.write("What you upload:")
	st.image(image_file, caption=500, width=500, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
	st.write(file_details["filename"])
	output_image_name=None
	change_bg.blur_bg(file_details["filename"], 
                  moderate = True , 
                  output_image_name="blur_img.jpg")

	st.write("What we allowed:")
	plt.imshow(Image.open("blur_img.jpg"))
	imggg="blur_img.jpg"
	st.image(imggg, caption=500, width=500, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
	count=0

