"""
This file contains the Predictor class, which is used to run predictions on the
Whisper model. It is based on the Predictor class from the original Whisper
repository, with some modifications to make it work with the RP platform.
"""

from beam import env

if env.is_remote():
    from faster_whisper.utils import format_timestamp
    import numpy as np


class Predictor:
    """ A Predictor class for the Whisper model """

    def __init__(self, model_instance):
        self.model = model_instance

    def predict(
        self,
        audio,
        transcription="plain_text",
        translate=False,
        translation="plain_text",
        language=None,
        temperature=0,
        best_of=5,
        beam_size=5,
        patience=1,
        length_penalty=None,
        initial_prompt=None,
        condition_on_previous_text=True,
        temperature_increment_on_fallback=0.2,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        enable_vad=False,
        word_timestamps=False
    ):
        if temperature_increment_on_fallback is not None:
            temperature = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
        else:
            temperature = [temperature]

        segments, info = list(self.model.transcribe(audio,
                                               language=language,
                                               task="transcribe",
                                               beam_size=beam_size,
                                               best_of=best_of,
                                               patience=patience,
                                               length_penalty=length_penalty,
                                               temperature=temperature,
                                               compression_ratio_threshold=compression_ratio_threshold,
                                               log_prob_threshold=logprob_threshold,
                                               no_speech_threshold=no_speech_threshold,
                                               condition_on_previous_text=condition_on_previous_text,
                                               initial_prompt=initial_prompt,
                                               prefix=None,
                                               suppress_blank=True,
                                               suppress_tokens=[-1],
                                               without_timestamps=False,
                                               max_initial_timestamp=1.0,
                                               word_timestamps=word_timestamps,
                                               vad_filter=enable_vad
                                               ))

        segments = list(segments)

        transcription = format_segments(transcription, segments)

        if translate:
            translation_segments, translation_info = self.model.transcribe(
                str(audio),
                task="translate",
                temperature=temperature
            )

            translation = format_segments(translation, translation_segments)

        results = {
            "segments": serialize_segments(segments),
            "detected_language": info.language,
            "transcription": transcription,
            "translation": translation if translate else None,
            "device": "cuda" if rp_cuda.is_available() else "cpu",
        }

        if word_timestamps:
            word_timestamps = []
            for segment in segments:
                for word in segment.words:
                    word_timestamps.append({
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                    })
            results["word_timestamps"] = word_timestamps


        return results


def serialize_segments(transcript):
    '''
    Serialize the segments to be returned in the API response.
    '''
    return [{
        "id": segment.id,
        "seek": segment.seek,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "tokens": segment.tokens,
        "temperature": segment.temperature,
        "avg_logprob": segment.avg_logprob,
        "compression_ratio": segment.compression_ratio,
        "no_speech_prob": segment.no_speech_prob
    } for segment in transcript]


def format_segments(format, segments):
    '''
    Format the segments to the desired format
    '''

    if format == "plain_text":
        return " ".join([segment.text.lstrip() for segment in segments])
    elif format == "formatted_text":
        return "\n".join([segment.text.lstrip() for segment in segments])
    elif format == "srt":
        return write_srt(segments)

    return write_vtt(segments)


def write_vtt(transcript):
    '''
    Write the transcript in VTT format.
    '''
    result = ""

    for segment in transcript:
        result += f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n"
        result += f"{segment.text.strip().replace('-->', '->')}\n"
        result += "\n"

    return result


def write_srt(transcript):
    '''
    Write the transcript in SRT format.
    '''
    result = ""

    for i, segment in enumerate(transcript, start=1):
        result += f"{i}\n"
        result += f"{format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')} --> "
        result += f"{format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')}\n"
        result += f"{segment.text.strip().replace('-->', '->')}\n"
        result += "\n"

    return result
