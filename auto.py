import datetime
import os
import random 
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import os
import sqlite3
from sqlite3 import Error
import pandas as pd
import sys
def messagetime(tx,ty,thx,thy,message):
    for i in range(2):
        minute =random.randrange(tx,ty)
        hour =random.randrange(thx,thy)
        txtfile = open("new.txt",'w')
        minute,hour = str(minute),str(hour)
        message =str(message)
        fa = "56 10 * * * python3.9 /Users/sophiasackstein/Desktop/python-automation/auto.py"+"\n"
        f = fa+minute +" "+hour+" * * * osascript Desktop/sendMessage.scpt 6464203504" +" \""+message+"\" "+"\n"
        txtfile.write(f)
        os.system("cat new.txt| crontab")


def load_tokenizer_and_model(model="microsoft/DialoGPT-large"): 
    # Initialize tokenizer and model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model)

    # Return tokenizer and model
    return tokenizer, model


def generate_response(sen, tokenizer, model, chat_round, chat_history_ids):
    """
    Generate a response to some user input.
    """
    # Encode user input and End-of-String (EOS) token
    new_input_ids = tokenizer.encode(sen + tokenizer.eos_token, return_tensors='pt')

    # Append tokens to chat history
    try:
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) 
    except:
        bot_input_ids = new_input_ids
        print("here!")

    # Generate response given maximum chat length history of 1250 tokens
    #chat_history_ids = model.generate(bot_input_ids, max_length=1250, pad_token_id=tokenizer.eos_token_id)
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1200,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        top_k=50, 
        top_p=0.9, 
        num_return_sequences=1,
        temperature=0.5
    )

    #print the output
    output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    # Print response
    print("AI: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

    # Return the chat history ids
    return output



def get_df():
    db_file = "/Users/sophiasackstein/Library/Messages/chat.db"
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    p= """ 
        select
     m.rowid,
     m.handle_id
    ,coalesce(m.cache_roomnames, h.id) ThreadId
    ,m.is_from_me IsFromMe
    ,case when m.is_from_me = 1 then m.account
     else h.id end as FromPhoneNumber
    ,case when m.is_from_me = 0 then m.account
     else coalesce(h2.id, h.id) end as ToPhoneNumber
    ,m.service Service

    /*,datetime(m.date + 978307200, 'unixepoch', 'localtime') as TextDate -- date stored as ticks since 2001-01-01 */
    ,datetime((m.date / 1000000000) + 978307200, 'unixepoch', 'localtime') as TextDate /* after iOS11 date needs to be / 1000000000 */

    ,m.text MessageText

    ,c.display_name RoomName

    from
    message as m
    inner join chat_message_join T2 on m.rowid=T2.message_id 
    inner join message T1 on m.rowid=T2.message_id 
    left join handle as h on m.handle_id = h.rowid
    left join chat as c on m.cache_roomnames = c.room_name /* note: chat.room_name is not unique, this may cause one-to-many join */
    left join chat_handle_join as ch on c.rowid = ch.chat_id
    left join handle as h2 on ch.handle_id = h2.rowid

    where (h2.service is null or m.service = h2.service) AND T2.chat_id=3

    order by m.date; """
    cur.execute(p)
    r = cur.fetchall()
    df_msg = pd.DataFrame(r, columns=['id_x', 'id_y','num','isme','numsend','numrecieve','type','date','text', 'attachments'])
    df = df_msg.drop_duplicates()
    return df 
def respondpls(df):
    v =0
    chat_round =-1
    #tokenizer, model = load_tokenizer_and_model()

    # Initialize history variable
    chat_history_idsy = None
    chat_history_idsx = None
    me =[]
    you =[]
    stopdf=len(df[df.isme==0])
    s = ""
    l=""
    for i,r in df.iterrows():
        chat_round+=1
        if r["isme"] == 1:

            me.append(r['text'])
            l = '. '.join(me)
            you =[]
            print("me: ",l)
            chat_history_idsy = generate_response(s,tokenizer, model, chat_round, chat_history_idsy)


        if r["isme"] == 0:

            me=[]
            #chat_history_idsx = generate_prev(l,tokenizer, model, chat_round, chat_history_idsy)
            you.append(r['text'])
            l = '. '.join(you)
            v+=1
            print("you: ", l)
            chat_history_idsy = generate_response(l,tokenizer, model, chat_round, chat_history_idsy)
            s=l
        if v==stopdf:
            chat_history_idsy = generate_response(s,tokenizer, model, chat_round, chat_history_idsy)
            return chat_history_idsy
        
      
    return chat_history_idsy

if __name__ == "__main__":
    mess = ["Getting out of work! stop worrying lol",
    "Hey! hope you're doing okay",
    "Almost of our work:)!",
    "How was your day?",
    "Still working.. sorry I can't call"]
    message=random.choice(mess)
    try:
        b= str(sys.argv[1])
        if b =="ai":
            df=get_df()
            response = respondpls(df)
        else:
            messagetime(tx=10,ty=59,thx=18,thy=22,message=message)
    except:
        
        messagetime(tx=10,ty=59,thx=18,thy=22,message=message)