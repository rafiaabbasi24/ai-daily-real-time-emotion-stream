import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from river import drift

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class EmotionStreamModel:
    """
    Wrapper for a multilingual emotion classifier (based on HuggingFace Transformers)
    and a River drift detector.

    The classifier returns probability scores for a predefined set of emotion labels.
    The drift detector (ADWIN by default) monitors the classification error rate in a
    streaming setting and signals when a significant change in the data distribution
    is observed.

    Example
    -------
    >>> model = EmotionStreamModel()
    >>> result = model.predict("I am so happy!")
    >>> result["label"]
    'joy'
    >>> # In a streaming loop
    >>> for text, true_label in stream:
    ...     pred = model.predict(text)
    ...     model.update(text, true_label)  # updates drift detector
    ...     if model.check_drift():
    ...         print("Concept drift detected!")
    """

    #: Default HuggingFace model that supports many languages.
    #: This model is trained on multilingual tweets and predicts six emotions.
    DEFAULT_MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-emotion"

    #: Emotion labels in the order returned by the model.
    EMOTION_LABELS = [
        "anger",
        "joy",
        "optimism",
        "sadness",
        "fear",
        "surprise",
    ]

    def __init__(
        self,
        model_name: str | None = None,
        device: str | torch.device | None = None,
        drift_detector: drift.base.DriftDetector | None = None,
    ) -> None:
        """
        Initialise the wrapper.

        Parameters
        ----------
        model_name: str | None
            Name or path of the HuggingFace model. If ``None`` the default multilingual
            model is used.
        device: str | torch.device | None
            Torch device to run inference on. If ``None`` automatically selects CUDA if
            available, otherwise CPU.
        drift_detector: river.drift.base.DriftDetector | None
            An instantiated River drift detector. If ``None`` an ADWIN detector with
            default parameters is created.
        """
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.device = self._resolve_device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.drift_detector = drift_detector or drift.ADWIN(delta=0.002)

        logger.info(
            "EmotionStreamModel initialised with model %s on device %s",
            self.model_name,
            self.device,
        )

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        """Resolve the torch device to use for inference."""
        if isinstance(device, torch.device):
            return device
        if isinstance(device, str):
            return torch.device(device)
        # Auto-select
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _preprocess(self, text: str) -> Dict[str, Tensor]:
        """
        Tokenise a single text string and move tensors to the appropriate device.

        Parameters
        ----------
        text: str
            Input text.

        Returns
        -------
        dict
            Dictionary with keys required by the model (e.g. ``input_ids``,
            ``attention_mask``).
        """
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Run inference on a single text string.

        Parameters
        ----------
        text: str
            Input text.

        Returns
        -------
        dict
            A dictionary containing:
            - ``label`` (str): the emotion with highest probability.
            - ``probabilities`` (dict): mapping from each emotion label to its
              probability (float).
            - ``logits`` (list[float]): raw model logits.
        """
        inputs = self._preprocess(text)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits: Tensor = outputs.logits.squeeze(0)  # shape: (num_labels,)

        probs = torch.softmax(logits, dim=0).cpu().tolist()
        label_idx = int(torch.argmax(logits, dim=0).item())
        label = self.EMOTION_LABELS[label_idx]

        result = {
            "label": label,
            "probabilities": dict(zip(self.EMOTION_LABELS, probs)),
            "logits": probs,
        }

        logger.debug("Prediction for text %r: %s", text, result)
        return result

    def update(self, text: str, true_label: str) -> None:
        """
        Update the drift detector with the classification error for a single instance.

        The error is defined as ``0`` if the predicted label matches ``true_label``,
        otherwise ``1``.

        Parameters
        ----------
        text: str
            Input text that was classified.
        true_label: str
            Ground‑truth emotion label (must be one of ``EMOTION_LABELS``).
        """
        if true_label not in self.EMOTION_LABELS:
            raise ValueError(
                f"Invalid true_label {true_label!r}. Expected one of {self.EMOTION_LABELS}."
            )

        pred = self.predict(text)
        error = 0 if pred["label"] == true_label else 1
        self.drift_detector.update(error)

        logger.debug(
            "Updated drift detector with error %d for true label %s (predicted %s)",
            error,
            true_label,
            pred["label"],
        )

    def check_drift(self) -> bool:
        """
        Check whether the drift detector has signalled a change.

        Returns
        -------
        bool
            ``True`` if drift is detected, ``False`` otherwise.
        """
        drifted = self.drift_detector.change_detected
        if drifted:
            logger.warning("Concept drift detected by River detector.")
        return drifted

    def reset_drift(self) -> None:
        """
        Reset the internal state of the drift detector.
        """
        self.drift_detector = drift.ADWIN(delta=0.002)
        logger.info("Drift detector has been reset.")

    def save(self, directory: str | Path) -> None:
        """
        Save the tokenizer, model and drift detector state to ``directory``.

        Parameters
        ----------
        directory: str | pathlib.Path
            Destination folder. It will be created if it does not exist.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        # Save transformer components
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        # Save drift detector (River objects are pickle‑compatible)
        drift_path = path / "drift_detector.pkl"
        with drift_path.open("wb") as f:
            import pickle

            pickle.dump(self.drift_detector, f)

        logger.info("Model and drift detector saved to %s", path)

    @classmethod
    def load(cls, directory: str | Path) -> "EmotionStreamModel":
        """
        Load a previously saved ``EmotionStreamModel`` from ``directory``.

        Parameters
        ----------
        directory: str | pathlib.Path
            Folder containing the saved model, tokenizer and drift detector.

        Returns
        -------
        EmotionStreamModel
            Re‑instantiated wrapper.
        """
        path = Path(directory)

        model_name = str(path)  # Load from local directory
        instance = cls(model_name=model_name)

        # Load drift detector
        drift_path = path / "drift_detector.pkl"
        if drift_path.is_file():
            with drift_path.open("rb") as f:
                import pickle

                instance.drift_detector = pickle.load(f)
            logger.info("Drift detector loaded from %s", drift_path)
        else:
            logger.warning(
                "Drift detector file %s not found; using a fresh ADWIN instance.", drift_path
            )

        return instance


__all__ = ["EmotionStreamModel"]