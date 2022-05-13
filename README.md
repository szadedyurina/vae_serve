# Image generation service

## How to:

<ol>
  <li> Install and run service

Make sure you have Docker installed on your machine.

```

git clone https://github.com/zvezdysima/vae_serve.git

cd vae_serve/server && docker-compose up --build
```

Open http://localhost:8000/ in your browser and enjoy the ultimate beautiful ugliness.

<li> Train model

Upload [notebook](https://github.com/zvezdysima/vae_serve/blob/main/vae_celeba.ipynb) to any server with GPU available and gdrive connected and just run all the cells. Generated samples are displayed in the notebooks, as well as model weights saved and LPIPS metric calcualted.
</ol>

## Model description

<ul>
<li> Model 

Model acrhitecture is briefly described in the [notebook](https://github.com/zvezdysima/vae_serve/blob/main/vae_celeba.ipynb). 
<li> Data

CelebA [dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) is used for training (about 200k images)
<li> Metrics
</ul>

[LPIPS](https://torchmetrics.readthedocs.io/en/v0.8.2/image/learned_perceptual_image_patch_similarity.html) metric is used to assess generated images quality 

## Service layout

There are the following components of the service:
 <ul>
 <li> Flask web server to handle requests
 <li> Celery app to handle async tasks
 <li> Redis message broker used by Celery app
 </ul>