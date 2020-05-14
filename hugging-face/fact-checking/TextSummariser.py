import os
import shutil

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from transformers import (
    BertTokenizer,
    TFBertForSequenceClassification,
)
from modeling_bertabs import BertAbs, build_predictor
from run_summarization import format_summary, collate, load_and_cache_examples
import tempfile
from argparse import Namespace

from utils_summarization import CNNDMDataset

model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'

"""
    --batch_size 4 
    --min_length 50
    --max_length 200
    --beam_size 5 
    --alpha 0.95 
    --block_trigram true 
    --compute_rouge true

"""


class TextSummariser(object):

    def __init__(self):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        symbols = {
            "BOS": tokenizer.vocab["[unused0]"],
            "EOS": tokenizer.vocab["[unused1]"],
            "PAD": tokenizer.vocab["[PAD]"],
        }
        model = BertAbs.from_pretrained("bertabs-finetuned-cnndm")
        args = Namespace()
        args.batch_size = 4
        args.beam_size = 5
        args.block_trigram = True
        args.compute_rouge = True
        args.device = "cpu"
        args.documents_dir = '/tmp/data/dataset1'
        args.max_length = 200
        args.min_length = 50
        args.no_cuda = True
        args.result_path = ''
        args.summaries_output_dir = '/Users/ryadhkhisb/Dev/workspaces/m/myelin-examples/hugging-face/fact-checking/summaries_out'
        args.temp_dir = ''

        args.alpha = 0.95
        #
        # args.beam_size = 5
        # args.min_length = 50
        # args.max_length = 200
        #
        # args.block_trigram = True
        #
        # args.dec_layers =
        # args.dec_hidden_size,
        # args.dec_heads,
        # args.dec_ff_size,
        # args.dec_dropout,
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.predictor = build_predictor(args, tokenizer, symbols, model)

    def predict(self, X, feature_names):
        dir_path = tempfile.mkdtemp()
        temp = open(os.path.join(dir_path, 'predict_data'), "w")
        temp.write(X)
        temp.close()
        dir_path = os.path.dirname(os.path.realpath(temp.name))

        dataset = CNNDMDataset(dir_path)
        sampler = SequentialSampler(dataset)

        def collate_fn(data):
            return collate(data, self.tokenizer, block_size=512, device=self.args.device)

        data_iterator = DataLoader(dataset, sampler=sampler, batch_size=1, collate_fn=collate_fn, )
        for batch in tqdm(data_iterator):
            batch_data = self.predictor.translate_batch(batch)
            translations = self.predictor.from_batch(batch_data)

        shutil.rmtree(dir_path)

        return [format_summary(t) for t in translations]


if __name__ == "__main__":
    m = TextSummariser()
    summary = m.predict("This is the end", {})
    print(summary)
    end = """
    That all our knowledge begins with experience there can be no doubt.
For how is it possible that the faculty of cognition should be awakened
into exercise otherwise than by means of objects which affect our
senses, and partly of themselves produce representations, partly rouse
our powers of understanding into activity, to compare to connect, or to
separate these, and so to convert the raw material of our sensuous
impressions into a knowledge of objects, which is called experience? In
respect of time, therefore, no knowledge of ours is antecedent to
experience, but begins with it.

But, though all our knowledge begins with experience, it by no means
follows that all arises out of experience. For, on the contrary, it is
quite possible that our empirical knowledge is a compound of that which
we receive through impressions, and that which the faculty of cognition
supplies from itself (sensuous impressions giving merely the occasion),
an addition which we cannot distinguish from the original element given
by sense, till long practice has made us attentive to, and skilful in
separating it. It is, therefore, a question which requires close
investigation, and not to be answered at first sight, whether there
exists a knowledge altogether independent of experience, and even of
all sensuous impressions? Knowledge of this kind is called a  priori, in
contradistinction to empirical knowledge, which has its sources a 
posteriori, that is, in experience.

But the expression, a  priori, is not as yet definite enough
adequately to indicate the whole meaning of the question above started.
For, in speaking of knowledge which has its sources in experience, we
are wont to say, that this or that may be known a  priori, because we do
not derive this knowledge immediately from experience, but from a
general rule, which, however, we have itself borrowed from experience.
Thus, if a man undermined his house, we say, œhe might know a  priori
that it would have fallen; that is, he needed not to have waited for
the experience that it did actually fall. But still, a  priori, he could
not know even this much. For, that bodies are heavy, and, consequently,
that they fall when their supports are taken away, must have been known
to him previously, by means of experience.

By the term knowledge a  priori, therefore, we shall in the sequel
understand, not such as is independent of this or that kind of
experience, but such as is absolutely so of all experience. Opposed to
this is empirical knowledge, or that which is possible only a 
posteriori, that is, through experience. Knowledge a  priori is either
pure or impure. Pure knowledge a  priori is that with which no empirical
element is mixed up. For example, the proposition, œEvery change has a
cause, is a proposition a  priori, but impure, because change is a
conception which can only be derived from experience.

    """
    # summary1 = m.predict(end, {})
    # print(summary1)

    txt = """
Wizz Air has said it will restart holiday flights from Luton airport to Portugal in mid-June and to Greece in July in the hope that Covid-19 travel restrictions will be lifted.

Announcing five new routes, the low-cost airline said from 16 June flights would depart from the London airport to Faro in Portugal, with prices starting at £25.99, and to Corfu, Heraklion, Rhodes and Zakynthos in Greece from the start of July.

Wizz Air has introduced new rules, including compulsory face masks for passengers and staff as well as gloves for crew, to make people feel more confident about flying.

It will also give sanitising wipes to travellers and no longer provide magazines. The airline says it encourages travellers to observe physical distancing at the airport but it will fill middle seats on aircrafts if there is enough demand.
    
Alexandre de Juniac, the director general of the International Air Transport Association, said during the group’s weekly briefing there was no evidence that leaving the middle seat empty would improve passenger safety.

Owain Jones, the managing director of Wizz Air UK, said: "Although travel is currently restricted by government regulations, we are planning for the easing of restrictions as the situation improves and our customers are able to start travelling again.”

UK nationals will be able to fly, assuming travel restrictions are lifted. The UK Foreign Office currently advises British nationals against all but essential international travel.

Portugal and Greece have started relaxing their lockdowns, although travel restrictions remain. Greece’s tourism minister, Haris Theoharis, said last month the sector hoped to be open for business in July.

European flights have come to a virtual standstill during the coronavirus lockdowns, with only a few services operating for essential travel, such as people being repatriated, going to work or to transport medical supplies.

Budapest-based Wizz Air became the first airline to resume commercial flights to and from Luton last Friday, and also restarted flights from Vienna. Most of the more than 100 passengers arriving at Luton from Sofia, Bulgaria, on a Wizz Air flight on Friday were seasonal farm workers.

The Dutch airline KLM also said on Tuesday that it had resumed operations to a number of its European destinations. It is now running a daily flight from Amsterdam to Barcelona, Budapest, Helsinki, Madrid, Milan, Prague, Rome and Warsaw.


    """
    summary1 = m.predict(txt, {})
    print(summary1)
    
    txt = """
As the headlines continues to be unrelentingly grim, there are still some stories out there to bring a smile to your face – from charity challenges to shows of support for the NHS.

Walking fast
Following the success of Capt Tom Moore’s fundraising effort, another centenarian has pledged to do laps of his garden to raise money for charity – but this time with an added challenge.

Dabirul Islam Choudhury is raising money for those affected by the virus in the UK and Bangladesh by walking laps of his community garden while fasting for Ramadan. The 100-year-old began walking 100 laps of the 80-metre garden on 26 April to raise £1,000, but hit the target within hours.

Since then Choudhury’s JustGiving page has raised more than £60,000 for the Ramadan Family Commitment (RFC) Covid-19 crisis initiative, run by British-Bangladeshi television broadcaster Channel S. He plans to continue fundraising for the entire month of Ramadan, which takes place this year from 23 April 23to 23 May, while continuing to observe religious fasting.

Choudhury was born on 1 January 1920 in British Assam, now modern-day Bangladesh, and moved to London to study English literature in 1957. His son, Atique Choudhury, told BBC London: "When we started, we started at a small pace but he’s been increasing his number of laps he’s doing. The problem we have is that we have to try and stop him because he wants to carry on.”

Paint the town blue
A teenager in Somerset has paid tribute to NHS staff by lighting up local landmark the Glastonbury Tor in blue. Inspired by a similar initiative at Windsor Castle, 18-year-old Sam Wardel used two small LED floodlights to illuminate the Grade 1 listed structure. He has so far staged his light show every two weeks, but now says he will do it every Thursday.

He told the BBC: "I had seen some houses and Windsor Castle had been lit up in blue… so I wanted to do something bigger to show my appreciation for everything the NHS has been doing.

"I just happened to look out of the kitchen window at the tower,” Wardel said. "I thought why don’t I go up there so everyone can see it for miles around.”

It’s in the delivery
Comedian Jason Manford has revealed he applied for a job at his local Tesco to help get groceries to people when the crisis first started. Tesco thanked him for his application, but said that since it came after the deadline, they couldn’t offer him the position.

Manford applied for a customer assistant position at the supermarket’s Aldery Road Express store in Wilmslow, Cheshire, when the chain advertised for more staff to help them with the increased demand during the pandemic in March.

He said he had applied as he felt it was "basically wartime and it would require all hands on deck”. The comedian joked on Twitter that he had even said on his CV: "Previous experience: Comedian. So I know that it’s all in the delivery!” He has since helped out doing volunteer driving for other organisations.

A spokesman for Tesco said: "We’ve recruited around 50,000 temporary workers during the coronavirus pandemic and they have played a huge part in helping us to serve customers safely during these unprecedented times.

"Jason’s skills would have no doubt brought a lot of joy to our customers and colleagues, so it’s a shame he didn’t make the deadline for this vacancy. But should he ever want to join the Tesco family in the future, we’d be happy to receive an application from him.”

Topics
Coronavirus outbreak
Hope in a time of crisis
Jason Manford
news
Reuse this content
    
    """
    summary1 = m.predict(txt, {})
    print(summary1)
