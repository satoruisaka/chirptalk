"""
chirp_eliza.py - chirptalk between two devices with Eliza-like conversation exchange

Satoru Isaka
December 27, 2019

A device sends a message via audio. Another device listens to the audio signal,
decodes the message, and sends a response message via audio.
The two devices continue to communicate with each other over sound.

Preparation:
Install python3 and pip3. Install mic and speaker.
Install Chirp SDK for data-over-sound function
Install NLTK for natural language chat function
Download the python code and run

Chirp SDK	https://developers.chirp.io/
NLTK: Natural Language Toolkit	https://www.nltk.org/
"""

# nltk components
from __future__ import print_function
import nltk
from nltk.chat.util import Chat, reflections

# chirpsdk components
import argparse
import sys
import time
from chirpsdk import ChirpSDK, CallbackSet, CHIRP_SDK_STATE

# chat data is a table of response pairs.
# Each pair consists of a regular expression and a list of responses
# with group-macros labelled as %1, %2.

pairs = (
    (
        r'I need (.*)',
        (
            "01234567890123456789012345678901",
            "Why do you need !@#?",
            "Would it really help you?",
            "Are you sure you need !@#?",
        ),
    ),
    (
        r'Why don\'t you (.*)',
        (
            "Do you think I don't !@#?",
            "Perhaps I will !@#.",
            "Do you want me to !@#?",
        ),
    ),
    (
        r'Why can\'t I (.*)',
        (
            "Do you think you can !@#?",
            "If you could, what would you do?",
            "I don't know, why can't you?",
            "Have you really tried?",
        ),
    ),
    (
        r'I can\'t (.*)',
        (
            "How do you know you can't !@#?",
            "Perhaps you could if you tried.",
            "What do you need to !@#?",
        ),
    ),
    (
        r'I am (.*)',
        (
            "Is that why you came to me?",
            "How long have you been !@#?",
            "How do you feel about that?",
        ),
    ),
    (
        r'I\'m (.*)',
        (
            "How does it feel beng !@#?",
            "Do you enjoy being !@#?",
            "Why do you tell me you're !@#?",
            "Why do you think you're !@#?",
        ),
    ),
    (
        r'Are you (.*)',
        (
            "Why does it matter to you?",
            "Do you prefer if I'm not that?",
            "Perhaps you think I am !@#.",
            "I may be !@#, what do you think?",
        ),
    ),
    (
        r'What (.*)',
        (
            "Why do you ask?",
            "How would an answer help you?",
            "What do you think?",
        ),
    ),
    (
        r'How (.*)',
        (
            "How do you suppose?",
            "Perhaps you can answer yourself",
            "What is it you're really asking?",
        ),
    ),
    (
        r'Because (.*)',
        (
            "Is that the real reason?",
            "What other reasons come to mind?",
            "Does it apply to anything else?",
            "If so, what else must be true?",
        ),
    ),
    (
        r'(.*) sorry (.*)',
        (
            "No apology is needed.",
            "How do you feel about that?",
        ),
    ),
    (
        r'Hello(.*)',
        (
            "Hello, I'm glad you drop by",
            "Hi there... how are you today?",
            "How are you feeling today?",
        ),
    ),
    (
        r'I think (.*)',
        (
            "Do you doubt !@#?",
            "Do you really think so?",
            "But you're not sure !@#?"
        ),
    ),
    (
        r'(.*) friend (.*)',
        (
            "Tell me more about your friends",
            "What comes to mind?",
            "Why don't you tell me more?",
        ),
    ),
    (
        r'Yes',
        (
            "You seem quite sure.",
            "OK, but can you elaborate a bit?",
        ),
    ),
    (
        r'(.*) computer(.*)',
        (
            "Are you talking about me?",
            "Does it seem strange to you?",
            "How do computers make you feel?",
            "Do you feel threatened?",
        ),
    ),
    (
        r'Is it (.*)',
        (
            "Do you think it is?",
            "Perhaps, what do you think?",
            "If so, what would you do?",
            "It could be !@#.",
        ),
    ),
    (
        r'It is (.*)',
        (
            "You seem very certain.",
            "If not, how would you feel?",
        ),
    ),
    (
        r'Can you (.*)',
        (
            "What makes you think I can't?",
            "If I could !@#, so what?",
            "Why do you ask?",
        ),
    ),
    (
        r'Can I (.*)',
        (
            "!@#?",
            "Do you want to !@#?",
            "If you could, would you?",
        ),
    ),
    (
        r'You are (.*)',
        (
            "Why do you think I am !@#?",
            "Does it please you to think so?",
            "Perhaps you like me to be so",
            "Are you talking about yourself?",
        ),
    ),
    (
        r'You\'re (.*)',
        (
            "!@#?",
            "You are !@#?",
            "Are we talking about you, or me?",
        ),
    ),
    (
        r'I don\'t (.*)',
        ("Don't you really !@#?", "Why don't you !@#?", "Do you want to !@#?"),
    ),
    (
        r'I feel (.*)',
        (
            "Good, tell me more",
            "Do you often feel that?",
            "When do you usually feel it?",
            "When you feel it, what do you do",
        ),
    ),
    (
        r'I have (.*)',
        (
            "Why do you have !@#?",
            "what, !@#?",
            "What will you do next?",
        ),
    ),
    (
        r'I would (.*)',
        (
            "Could you explain why?",
            "Why would you?",
            "Who else knows that?",
        ),
    ),
    (
        r'Is there (.*)',
        (
            "what, !@#?",
            "It's likely",
            "Would you like about it?",
        ),
    ),
    (
        r'My (.*)',
        (
            "I see, !@#.",
            "Why !@#?",
            "How do you feel?",
        ),
    ),
    (
        r'You (.*)',
        (
            "Let's talk about ou, not me.",
            "Why do you care?",
            "You are !@#?",
        ),
    ),
    (r'Why (.*)', ("Why don't you tell me why?", "Why do you think so?")),
    (
        r'I want (.*)',
        (
            "What would it mean to you?",
            "Why do you want it?",
            "What would you do if you had it?",
            "What, !@#?",
        ),
    ),
    (
        r'(.*) mother(.*)',
        (
            "Tell me more about your mother.",
            "How is your mom?",
            "How do you feel about your mom?",
            "How does it feel?",
            "Good family is important.",
        ),
    ),
    (
        r'(.*) father(.*)',
        (
            "Tell me more about your dad.",
            "How does your dad make you feel?",
            "How do you feel about your dad?",
            "Dad?",
            "Do you have trouble with dad?",
        ),
    ),
    (
        r'(.*) child(.*)',
        (
            "Did you have a child?",
            "What is your childhood memory?",
            "Do you remember any dreams?",
            "Did other children tease you?",
            "What do you think?",
        ),
    ),
    (
        r'(.*)\?',
        (
            "Why do you ask that?",
            "Please think.",
            "The answer lies within yourself?",
            "Why don't you tell me?",
        ),
    ),
    (
        r'quit',
        (
            "Thank you for talking with me.",
            "Good-bye.",
            "Thank you, Have a good day!",
        ),
    ),
    (
        r'(.*)',
        (
            "Please tell me more.",
            "Let's talk about your feelings.",
            "Can you elaborate on that?",
            "Why !@#?",
            "I see.",
            "Very interesting.",
            "!@#.",
            "I see. What does that tell you?",
            "How does that make you feel?",
            "How does it feel saying that?",
        ),
    ),
)

# create an eliza-like chat
eliza_chatbot = Chat(pairs, reflections)


# chirptalk segment

rdata = bytearray(32)
payloadlength = 0

class Callbacks(CallbackSet):

    def on_state_changed(self, previous_state, current_state):
        """ Called when the SDK's state has changed """
#        print('State changed from {} to {}'.format(
#            CHIRP_SDK_STATE.get(previous_state),
#            CHIRP_SDK_STATE.get(current_state)))

    def on_sending(self, payload, channel):
        """ Called when a chirp has started to be transmitted """
#        print('Sending: {data} [ch{ch}]'.format(
#            data=list(payload), ch=channel))

    def on_sent(self, payload, channel):
        """ Called when the entire chirp has been sent """
#        print('Sent: {data} [ch{ch}]'.format(
#            data=list(payload), ch=channel))

    def on_receiving(self, channel):
        """ Called when a chirp frontdoor is detected """
#        print('Receiving data [ch{ch}]'.format(ch=channel))

    def on_received(self, payload, channel):
        """
        Called when an entire chirp has been received.
        Note: A payload of None indicates a failed decode.
        """
        global payloadlength


        if payload is None:
            print('Decode failed!')
        else:
#            print('Received: {data} [ch{ch}]'.format(
#                data=list(payload), ch=channel))
# load up a bytearray "rdata[]" with the received payload
            payloadlength = len(payload)
#            print(payloadlength)
            i = 0
            for x in payload:
               rdata[i] = x
               i+=1

def main(block_name, input_device, output_device,
         block_size, sample_rate, channel):

    global payloadlength

    # Initialise Chirp SDK
    sdk = ChirpSDK(block=block_name)
    print(str(sdk))
    print('Protocol: {protocol} [v{version}]'.format(
        protocol=sdk.protocol_name,
        version=sdk.protocol_version))
    print(sdk.audio.query_devices())

    # Configure audio
    sdk.audio.input_device = input_device
    sdk.audio.output_device = output_device
    sdk.audio.block_size = block_size
    sdk.input_sample_rate = sample_rate
    sdk.output_sample_rate = sample_rate

    # Set callback functions
    sdk.set_callbacks(Callbacks())

    # Set transmission channel for multichannel protocols
    if args.channel is not None:
        if args.channel >= sdk.channel_count:
            raise ValueError('Channel %d is not available' % args.channel)
#        print('Writing to channel %d' % args.channel)
        sdk.transmission_channel = args.channel

    # Send a message
    # [we don't do random payload in this code] Generate random payload and send
    #    payload = sdk.random_payload()
    # start from the user-supplied message in main args
    message = args.message.encode('utf-8')
    payload = sdk.new_payload(message)
    sdk.start(send=True, receive=True)
    sdk.send(payload)

    tom0 = 0
    waittime = 0

    try:
        # Process audio streams
        while True:
            tom = sdk.state
            if (tom == 2) & (tom0 == 4):
#                print('CHIRP RECEIVED')
                i = 0
# setup a new payload bytearray "pdata[]" 
                pdata = bytearray(payloadlength)
#                print(payloadlength)
                for i in range(payloadlength):
                  pdata[i] = rdata[i]
                msg = pdata.decode('utf-8')
                print('Received: {data}'.format(data=msg))
#                print(msg)

# code segment here to handle message response
# first, send the received message to chat handler
# and receive a response message
                response = eliza_chatbot.respond(msg)
                print('Response: {data}'.format(data=response))
# encode the response message
                newmsg = response.encode('utf-8')
# load up the payload with the encoded message
                payload = sdk.new_payload(newmsg)
# send the payload
                time.sleep(2)
                sdk.send(payload)
#                sdk.send(pdata)
                waittime = 0

            time.sleep(0.1)
#            sys.stdout.write('.')
            sys.stdout.flush()
            tom0 = tom
            waittime += 1

# if no response for a long time (30 sec), ping if anyone is listening
            if waittime > 300:
                response = "Hello, anyone there?"
                print('Response: {data}'.format(data=response))
                newmsg = response.encode('utf-8')
                payload = sdk.new_payload(newmsg)
                sdk.send(payload)
                waittime = 0

    except KeyboardInterrupt:
        print('Exiting')

    sdk.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ChirpSDK Demo',
        epilog='Sends a random chirp payload, then continuously listens for chirps'
    )
    parser.add_argument('message', help='Text message to send')
    parser.add_argument('-c', '--channel', type=int, help='The channel to output data on')
    parser.add_argument('-i', '--input-device', type=int, default=None, help='Input device index (optional)')
    parser.add_argument('-o', '--output-device', type=int, default=None, help='Output device index (optional)')
    parser.add_argument('-b', '--block-size', type=int, default=0, help='Block size (optional)')
    parser.add_argument('-s', '--sample-rate', type=int, default=44100, help='Sample rate (optional)')
    parser.add_argument('--config', type=str, help='The configuration block [name] in your ~/.chirprc file (optional)')
    parser.add_argument('--network-config', action='store_true', help='Optionally download a config from the network')
    args = parser.parse_args()

    main(args.config, args.input_device, args.output_device, args.block_size, args.sample_rate, args.channel)
