import json
import os
from typing import List, Dict, Any


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a dataset from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        List of dictionaries containing tasks and reference outputs
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def save_dataset(dataset: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save a dataset to a JSON file.

    Args:
        dataset: List of dictionaries to save
        file_path: Path to save the JSON file
    """
    if not file_path.endswith('.json'):
        raise ValueError('File path must end with .json')
    
    # Create directory if it doesn't exist
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(dataset, f, indent=2)


def create_example_dataset() -> List[Dict[str, Any]]:
    """
    Create an example dataset for testing purposes.

    Returns:
        An example dataset
    """
    return [
        {
            "task": "Write an email to a colleague about scheduling a meeting",
            "reference_output": "Hey Sam,\n\nHope you're good! Do you have time for a quick meeting next week? I want to go over the project timeline. Maybe Tuesday or Wednesday afternoon?\n\nLet me know what works!\n\nThanks,\nAlex"
        },
        {
            "task": "Write a short blog post about artificial intelligence",
            "reference_output": "AI is changing everything. I've been watching this space for years and honestly, it's moving faster than anyone expected. The tools we have today would've seemed like sci-fi just 5 years ago.\n\nWhat's wild is how it's seeping into our everyday lives. From the emails we write to how we search for info - AI is there, quietly helping.\n\nBut here's the thing - we're just getting started. The next few years? Mind-blowing stuff coming. I can't wait!"
        },
        {
            "task": "Write a product review for wireless headphones",
            "reference_output": "Just got these headphones last week. First impression? They're super comfortable! Been wearing them for hours and no ear pain (big deal for me).\n\nSound quality is pretty awesome too. Clear, nice bass, not too overwhelming. Battery life is solid - about 6 hours before needing a charge.\n\nOnly downside is the mic. It's ok for calls, but if you're in a noisy place, people struggle to hear you.\n\nOverall though? Totally worth the money. 8/10 would recommend!"
        }
    ]
