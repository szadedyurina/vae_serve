import base64
from celery import Celery
from celery.result import AsyncResult
from flask import Flask, request, render_template
from io import BytesIO
import matplotlib.pyplot as plt
import os
import torch
from torchvision.utils import make_grid

from model import VAE

device = 'cuda' if torch.cuda.is_available() else 'cpu'
PATH = './vae_celeba_model'
REDIS_HOST = os.environ['REDIS_HOST']

celery_app = Celery('server', backend=f'redis://{REDIS_HOST}', broker=f'redis://{REDIS_HOST}')
app = Flask(__name__)

def load_model(PATH):
    vae_model = VAE()
    vae_model.load_state_dict(torch.load(PATH, map_location=device))
    vae_model.eval()
    return vae_model

model = load_model(PATH)

@celery_app.task
def sample(size):
    with torch.no_grad():
      # generate random sample z from prior p(z) and pass through the decoder
        mu, sigma = torch.zeros((size, model.nz)).to(device), torch.ones((size, model.nz)).to(device)
        z = model.reparametrize(mu, sigma)
        dec_output = model.decode(z).cpu()
        log_img = make_grid(dec_output)
    
    # transform generated tensor into image to be rendered
    plt.figure(figsize=(3,3))
    plt.imshow(log_img.permute(1,2,0))  
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    plot_url = base64.b64encode(figfile.getvalue()).decode()
    plt.gcf().clear()
    return plot_url

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/generate')
def sample_handler():
    task = sample.delay(1)
    plot_url = task.wait()
    return render_template('index.html', image_url=plot_url)

if __name__ == '__main__':
    app.run("0.0.0.0", 8000)