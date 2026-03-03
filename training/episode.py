import torch

from data.arithmetic import extract_number


def run_episode(base_model, mach, patched_model, tokenizer, problems, device):
    """
    One meta-training episode.

    Episode = sequence of arithmetic problems.
    Each problem: observe → fire → write patches → forward Qwen → get reward.
    Patches accumulate writes across the episode.

    Returns:
        total_loss: differentiable, for meta-training backprop
        rewards: list of reward floats (for logging/reward signals)
        problem_losses: list of per-problem loss values (for diagnostics)
    """
    mach.reset_episode()

    rewards = []
    problem_losses = []
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    last_reward = 0.0
    cumulative_reward = 0.0

    for i, problem in enumerate(problems):
        input_ids = tokenizer(
            problem["prompt"], return_tensors="pt"
        ).input_ids.to(device)

        # Step 1: Observe (detached — GRU integrates Qwen hidden state)
        gru_memory = mach.observe(base_model, input_ids)

        # Step 2: Fire meta-learner
        reward_signals = torch.tensor(
            [last_reward, cumulative_reward, float(i)],
            device=device, dtype=torch.float32
        )
        writes = mach.fire(gru_memory, reward_signals)

        # Step 3: Apply writes (differentiable — delta_W stays in graph)
        mach.apply_writes(writes)

        # Step 4: Forward Qwen WITH modified patches, compute loss
        full_text = problem["prompt"] + problem["answer"]
        encoding = tokenizer(full_text, return_tensors="pt").to(device)
        prompt_len = len(tokenizer(problem["prompt"]).input_ids)
        labels = encoding.input_ids.clone()
        labels[0, :prompt_len] = -100

        output = patched_model(input_ids=encoding.input_ids, labels=labels)

        # Step 5: Compute reward from logits (no extra generate call)
        with torch.no_grad():
            logits = output.logits
            # Get argmax predictions at answer token positions
            pred_tokens = logits[0, prompt_len - 1:-1].argmax(dim=-1)
            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
            predicted = extract_number(pred_text)
            correct = (predicted == problem["answer"])
            reward = 1.0 if correct else -1.0

        rewards.append(reward)
        last_reward = reward
        cumulative_reward += reward
        problem_losses.append(output.loss.item())

        # Accumulate loss (differentiable path)
        total_loss = total_loss + output.loss

    return total_loss, rewards, problem_losses


def run_episode_chunked(base_model, mach, patched_model, tokenizer, problems,
                        device, optimizer):
    """
    Memory-saving fallback: backward per problem instead of full-episode.
    Gives gradient through current problem's writes but not across problems.
    Use if full-episode backprop exceeds VRAM.
    """
    mach.reset_episode()

    rewards = []
    problem_losses = []
    last_reward = 0.0
    cumulative_reward = 0.0
    total_loss_value = 0.0

    for i, problem in enumerate(problems):
        input_ids = tokenizer(
            problem["prompt"], return_tensors="pt"
        ).input_ids.to(device)

        # Observe (detached)
        gru_memory = mach.observe(base_model, input_ids)

        # Fire
        reward_signals = torch.tensor(
            [last_reward, cumulative_reward, float(i)],
            device=device, dtype=torch.float32
        )
        writes = mach.fire(gru_memory, reward_signals)

        # Apply writes
        mach.apply_writes(writes)

        # Forward + loss
        full_text = problem["prompt"] + problem["answer"]
        encoding = tokenizer(full_text, return_tensors="pt").to(device)
        prompt_len = len(tokenizer(problem["prompt"]).input_ids)
        labels = encoding.input_ids.clone()
        labels[0, :prompt_len] = -100

        output = patched_model(input_ids=encoding.input_ids, labels=labels)

        # Per-problem backward (accumulates gradients)
        output.loss.backward()

        # Reward from logits
        with torch.no_grad():
            logits = output.logits
            pred_tokens = logits[0, prompt_len - 1:-1].argmax(dim=-1)
            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
            predicted = extract_number(pred_text)
            correct = (predicted == problem["answer"])
            reward = 1.0 if correct else -1.0

        rewards.append(reward)
        last_reward = reward
        cumulative_reward += reward
        problem_losses.append(output.loss.item())
        total_loss_value += output.loss.item()

    return total_loss_value, rewards, problem_losses
