import json

import torch


def average_tool_embeddings(
    episodes_path: str,
) -> dict[str, torch.Tensor]:
    embeddings_by_tool = {}
    with open(episodes_path, encoding='utf-8') as episodes_file:
        for line in episodes_file:
            result = json.loads(line)
            embedding = torch.load(f=result['embedding_path']).float()
            tool_name = result['expected_tool_name']
            if tool_name not in embeddings_by_tool:
                embeddings_by_tool[tool_name] = []
            embeddings_by_tool[tool_name].append(embedding)

    return {
        tool_name: torch.stack(tensors=embeddings).mean(dim=0)
        for tool_name, embeddings in embeddings_by_tool.items()
    }


def get_tiktok_vector(
    tool_embeddings: dict[str, torch.Tensor],
) -> torch.Tensor:
    return tool_embeddings['watch_tiktok']


def get_random_vector(
    tool_embeddings: dict[str, torch.Tensor],
    seed: int = 42,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        size=tool_embeddings['watch_tiktok'].shape,
        generator=generator,
    )


def get_tiktok_minus_others_vector(
    tool_embeddings: dict[str, torch.Tensor],
) -> torch.Tensor:
    other_tool_vectors = [
        embedding
        for tool_name, embedding in tool_embeddings.items()
        if tool_name != 'watch_tiktok'
    ]
    return tool_embeddings['watch_tiktok'] - torch.stack(tensors=other_tool_vectors).mean(dim=0)


def get_negative_others_vector(
    tool_embeddings: dict[str, torch.Tensor],
) -> torch.Tensor:
    other_tool_vectors = [
        embedding
        for tool_name, embedding in tool_embeddings.items()
        if tool_name != 'watch_tiktok'
    ]
    return -torch.stack(tensors=other_tool_vectors).mean(dim=0)
