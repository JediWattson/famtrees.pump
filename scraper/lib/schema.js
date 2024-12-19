const GlobalSchema = {
	struct: {
		initialized: 'bool',
		authority: { array: { type: 'u8', len: 32 }},
		feeRecipient: { array: { type: 'u8', len: 32 }},
		initialVirtualTokenReserves: 'u64',
		initialVirtualSolReserves: 'u64',
		initialRealTokenReserves: 'u64',
		tokenTotalSupply: 'u64',
		feeBasisPoints: 'u64',
	}
}

const BondingCurveSchema = {
	struct: {
		virtualTokenReserves: 'u64',
		virtualSolReserves: 'u64',
		realTokenReserves: 'u64',
		realSolReserves: 'u64',
		tokenTotalSupply: 'u64',
		complete: 'bool',
	}
}

const TradeEventSchema = {
	struct: {
		mint: { array: { type: 'u8', len: 32 }},
		solAmount: 'u64',
		tokenAmount: 'u64',
		isBuy: 'bool',
		user: { array: { type: 'u8', len: 32 }},
		timestamp: 'i64',
		virtualSolReserves: 'u64',
		virtualTokenReserves: 'u64'
	}

}

module.exports = {
	BondingCurveSchema,
	GlobalSchema,
	TradeEventSchema
}
