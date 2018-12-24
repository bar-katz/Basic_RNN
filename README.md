# Basic LSTM on regular expression languages
> Create a regular expression language and run the LSTM.

This project is made in order to test LSTM capabilities.

## Usage example

1. Generate train and dev sets.<br> `gen_examples.py` is an example on how to create a language.<br><br>
The script `gen_examples.py` will create regular expression samples for the language:<br>
`[0-9]+a+[0-9]+b+[0-9]+c+[0-9]+d+[0-9]+`<br><br>
Parameters:<br>
`samples_file` – the file which the samples will be writen to.<br>
`num_samples` – number of samples to generate.<br>
`seq_max` – maximum length of each contiguous sequence, for example a+ will be a sequence of random number of `a`<br>in range [1, `seq_max`].<br><br>example:
    ```sh
    python gen_examples.py data/train 1000 50
    ```
2. Train the LSTM model on the language.<br>
Run `basicLSTM.py` on the data you created and check the results.<br><br>
Parameters:<br>
`train_samples` – train samples file.<br>
`dev_samples` – dev samples file.<br><br>
example:
    ```sh
    python basicLSTM.py data/train data/dev
    ```
## Build With
* [PyTorch](https://pytorch.org/docs/stable/index.html) – the deep learning platform used


## Author

Bar Katz – [bar-katz on github](https://github.com/bar-katz) – barkatz138@gmail.com

## Contributing

1. Fork it (<https://github.com/bar-katz/Basic_RNN/fork>)
2. Create your feature branch (`git checkout -b feature`)
3. Commit your changes (`git commit -am 'add feature'`)
4. Push to the branch (`git push origin feature`)
5. Create a new Pull Request
<br>
##### Share your results in the comments!
