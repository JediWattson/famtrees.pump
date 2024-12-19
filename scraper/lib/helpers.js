const { PublicKey } = require('@solana/web3.js')
const fs = require('fs')

const getPrice = ({ virtualSolReserves, virtualTokenReserves }) => {
	const priceInLamports = 
		BigInt(virtualSolReserves) * BigInt(1e9) / BigInt(virtualTokenReserves);
	return Number(priceInLamports) / 1e9;
}

const makeQueriedKey = () => {
	const pumpId = '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P'
	return new PublicKey(pumpId);
}

const makePubKey = (arr) => {
	const uArr = new Uint8Array(arr)
	return new PublicKey(uArr)
}

const makeBuffer = (dataType, data) => {
	const errDataType = dataType.verify(data)
	if (errDataType) throw errDataType
	const message = dataType.create(data)
	const buffer = dataType.encode(message).finish()

	const typeBuffer = Buffer.alloc(1);
	typeBuffer.writeInt8(data.tokenizedLogs ? 0 : 1, 0)

    const sizeBuffer = Buffer.alloc(4);
    sizeBuffer.writeUInt32BE(buffer.length, 0);

	fs.appendFileSync('../data.bin', Buffer.concat([sizeBuffer, typeBuffer, buffer]))
}

module.exports = {
	makePubKey,
	getPrice,
	makeBuffer,
	makeQueriedKey
}
