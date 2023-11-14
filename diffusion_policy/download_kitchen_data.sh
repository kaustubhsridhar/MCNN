mkdir data && cd data
wget https://diffusion-policy.cs.columbia.edu/data/training/kitchen.zip
unzip kitchen.zip && rm -f kitchen.zip && cd ..
wget -O kitchen_diffusion_policy_cnn.yaml https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/kitchen/diffusion_policy_cnn/config.yaml

