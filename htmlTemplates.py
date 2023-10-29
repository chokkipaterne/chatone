css = '''
<style>
/* feedback checkbox */
.css-18fuwiq {
 position: relative;
 padding-top: 6px;
}
.css-949r0i {
 position: relative;
 padding-top: 6px;
}

/* chat */
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
'''

css_chat = '''
<style>
body, html {
    background-color: #fff;
}
.block-container{
    max-width: 90% !important;
}

/* feedback checkbox */
.css-18fuwiq {
 position: relative;
 padding-top: 6px;
}
.css-949r0i {
 position: relative;
 padding-top: 6px;
}

.chat-row {
    display: flex;
    margin: 5px;
    width: 100%;
}

.row-reverse {
    flex-direction: row-reverse;
}

.chat-bubble {
    font-family: "Source Sans Pro", sans-serif, "Segoe UI", "Roboto", sans-serif;
    border: 1px solid transparent;
    padding: 5px 10px;
    margin: 0px 7px;
    max-width: 70%;
}

.ai-bubble {
    background: rgb(240, 242, 246);
    border-radius: 10px;
    color: black;
}

.human-bubble {
    background: linear-gradient(135deg, rgb(0, 178, 255) 0%, rgb(0, 106, 255) 100%); 
    color: white;
    border-radius: 20px;
}

.chat-icon {
    border-radius: 5px;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://www.pngall.com/wp-content/uploads/12/Avatar-Profile-PNG-Images.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''