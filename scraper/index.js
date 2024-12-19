const { Connection, clusterApiUrl } = require('@solana/web3.js')
const { deserialize } = require('borsh')

const { upsert } = require('./lib/proto')
const { makePubKey, getPrice, makeQueriedKey } = require('./lib/helpers')
const { BondingCurveSchema, TradeEventSchema } = require('./lib/schema')
const Tokenizer = require('./lib/tokenizer')

const tokenizer = new Tokenizer()
const handleLog = ({ logs }, { slot }) => {
	const tokenizedLogs = logs.map((log) => {
		const tokes = tokenizer.tokenizeLine(log)
		if (!log.includes('data:')) return tokes
	
		const data = Buffer.from(log.split(' ')[2])
		try {
			const trade = deserialize(TradeEventSchema, data)
			const mint = makePubKey(trade.mint).toBase58()
			const mintId = tokenizer.tokenizeLine(mint, true)
			const priceInSol = getPrice(trade)
			upsert({ mintId, priceInSol, slot })
		} catch (err) {
			console.log(log)
			return tokes
		}
		return tokes
	}).flat()

	upsert({ slot, tokenizedLogs })
}

const handleProgram = (info, { slot }) => {
	const data = Buffer.from(info.accountInfo.data)
	try {
		const curve = deserialize(BondingCurveSchema, data)
		const mint = info.accountId.toBase58()
		const mintId = tokenizer.tokenizeLine(mint, true)
		const priceInSol = getPrice(curve)
		upsert({ mintId, priceInSol, slot })
	} catch (err) {
		console.log(data.toString())
	}
}


const start = () => {
	const connection = new Connection(
		clusterApiUrl('mainnet-beta'), 
		'confirmed'
	)

	const pumpFunProgramId = makeQueriedKey()
	connection.onLogs(pumpFunProgramId, handleLog, 'confirmed')
	connection.onProgramAccountChange(pumpFunProgramId, handleProgram)
}

start()
process.stdin.resume(); 

