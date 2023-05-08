defmodule Eml.ZeroShot do
  alias Bumblebee
  alias Nx

  @model "facebook/bart-large-mnli"
  @model_spec_module Bumblebee.Text.Bart
  @tokenizer_module Bumblebee.Text.BartTokenizer
  @architecture :for_sequence_classification

  def classify_serving(labels) do
    {:ok, model_info} =
      Bumblebee.load_model({:hf, @model},
        module: @model_spec_module,
        architecture: @architecture
      )

    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, @model}, module: @tokenizer_module)

    Bumblebee.Text.zero_shot_classification(model_info, tokenizer, labels)
  end

  def classify_this(labels, sentence) do
    serving = classify_serving(labels)
    %{predictions: scores} = Nx.Serving.run(serving, sentence)

    scores
    |> Enum.max_by(fn label -> label.score end)
    |> then(& &1.label)
  end

  def classification_loop(labels) do
    serving = classify_serving(labels)
    classification_loop(serving, labels)
  end

  defp classification_loop(serving, labels) do
    classify_this = IO.gets("Type a sentence you would like to classify.")
    IO.inspect(Nx.Serving.run(serving, classify_this), label: "OUTPUT: ")

    case IO.gets("Continue? y/n") do
      "y" -> classification_loop(serving, labels)
      "n" -> "Goodbye!"
      _ -> "I don't understand."
    end
  end
end
