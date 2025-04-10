from django import forms


class CommentForm(forms.Form):
    content = forms.CharField(widget=forms.TextInput())
    mode = forms.ChoiceField(choices=(('lstm2x', 'lstm2x'), ('biGRU', 'biGRU'),('distilBERT','distilBERT')))


