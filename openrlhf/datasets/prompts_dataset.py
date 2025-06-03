from torch.utils.data import Dataset
from tqdm import tqdm


def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None, tokenizer=None) -> str:
    if apply_chat_template:
        raise NotImplementedError("apply_chat_template is not implemented for satori.")
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        # add bos token, important!
        if tokenizer.bos_token:
            prompt = tokenizer.bos_token + prompt

        if input_template:
            prompt = input_template.format(prompt)
        
        answer = data["answer"]
        reflect_status = data.get("reflect_status", 0)

        if "reflect_status" not in data:
            print("[Warning] reflect_status not found in data. Use 0 as default.")

    return prompt, answer, reflect_status


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        self.answers = []
        self.reflect_statuses = []
        self.datasources = []

        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt, answer, reflect_status = preprocess_data(data, input_template, input_key, apply_chat_template, tokenizer)
            self.prompts.append(prompt)
            self.answers.append(answer)
            self.reflect_statuses.append(reflect_status)
            self.datasources.append(data.get("datasource", "default"))

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return (
            self.datasources[idx],
            self.prompts[idx],
            self.answers[idx],
            self.reflect_statuses[idx],
        )
