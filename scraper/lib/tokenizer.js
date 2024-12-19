class SubwordTokenizer {
    constructor() {
        this.wordVocabulary = {
            'Program': 1,
            'invoke': 2,
            'Instruction': 3,
            'consumed': 4,
            'success': 5
        };

        this.charVocabulary = {};
        for (let i = 32; i <= 126; i++) {
            this.charVocabulary[String.fromCharCode(i)] = i - 24; 
        }
		this.charVocabulary['\n'] = 127
        this.charVocabulary['UNKNOWN_CHAR'] = 128; 
    }

    splitUnknownWord(word) {
		return Array.from(word)
			.map(char => this.charVocabulary[char] || this.charVocabulary['UNKNOWN_CHAR']);
    }

	tokenizeLine(line, isNoNewline) {
        let tokens = [];
        let words = line.split(/\s+/);
		let count = 0
        for (let word of words) {
            if (this.wordVocabulary[word]) {
                tokens.push(this.wordVocabulary[word])
            } else {
                tokens.push(...this.splitUnknownWord(word))
            }
			if (++count === words.length) break;	
			tokens.push(this.charVocabulary[' '])
        }

		if (!isNoNewline)
			tokens.push(this.charVocabulary['\n'])
        
		return tokens;
    }

	detokenizeLine(tokens) {
		const inverseWords = Object.entries(this.wordVocabulary)
			.reduce((acc, [k, v]) => ({ ...acc, [v]: k }), {})
		const inverseTexts = Object.entries(this.charVocabulary)
			.reduce((acc, [k, v]) => ({ ...acc, [v]: k }), {})

		return tokens.map(token => {
			if (token < 6 && token > 0)
				return inverseWords[token]
			return inverseTexts[token]
		}).join('')
	}
}

module.exports = SubwordTokenizer
