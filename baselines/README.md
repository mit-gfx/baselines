# Express test
1. Navigate to baselines/baselines.
2. I assume your Python is from Anaconda3 or something equivalent.
3. Run:
```bash
python run_pareto_ppo1.py --model-path tmp/walker --env Walker2d-v2 --num-timesteps 10000
```
This will train a neural network controller for the Walker2d example using PPO. The log will be stored in files in a `tmp` folder with prefix `walker`. Increasing the num-timesteps will give you a better trained controller.
4. To test the trained network controller, type:
```bash
python run_pareto_ppo1.py --model-path tmp/walker --env Walker2d-v2 --num-timesteps 1000 --play
```
Here the number after num-timesteps does not matter. It will open a MuJoCo window that renders the results. The rendering is restarted whenever a single rollout is terminated.
5. A summary of all the working environments:
- InvertedDoublePendulum: continuous states and continuous actions.
- Hopper: continuous states and continuous actions.