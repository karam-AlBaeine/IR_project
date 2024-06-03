from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# # download model 
# model_name = "facebook/bart-large-cnn"
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # save model
# model.save_pretrained("./local_model")
# tokenizer.save_pretrained("./local_model")


def summarize_text(text, max_length=130, min_length=30):
    # load model
    model = AutoModelForSeq2SeqLM.from_pretrained("./local_model")
    tokenizer = AutoTokenizer.from_pretrained("./local_model")
    
    # create a summarize text
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# text = 'in 1996 he moved to glasgow to join the first ever intake of the ba scottish music course at the rsamd now called the royal conservatoire of scotland he graduated from the rsamd in 1999 playing with a series of scottish traditional music bands in glasgow in 2000 he had been working as margaret bennett s accompanist and had toured with her in scotland and france after working on the in the sunny long ago with producer martyn bennett he was invited by gillian frame hamish napier and simon mckerrell to join back of the moon that year back of the moon recorded their debut album gillian frame and back of the moon back of the moon toured extensively from 2000 till 2007 releasing three albums and finishing with a final gig in the kennedy centre in washington d c on 21 november 2007 while recording back of the moon s second album at watercolour music in ardgour findlay was approached by producer and engineer nick turner to begin a song writing project which they later named queen anne s revenge they began writing on the evening of 14 december 2003 nick s 42nd birthday and had written four songs'

# print(summarize_text(text))