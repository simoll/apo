import sys
from matplotlib import pyplot as pl

logFile = sys.argv[1]

# - Round 200 (global_step=199, learning_rate=0.001) -
def parseRoundHeader(l):
  parts = l.split(' ')
  trainRound = int(parts[2]) # '' 'Round' '200'
  return trainRound

# Greedy (best)  0.8790  (matched 0.8550, longerDer 0.0020, shorterDer: 0.0030, betterScore: 0.0190)
def parseEvalLine(l):
  # "best)  0.8790  "
  parts = l.split("matched")
  leftParts = parts[0][:-1].strip().split(" ")
  totalText = leftParts[-1] 

  detailParts = parts[1][:-1].split(' ')
  betterText = detailParts[-1][:-1]
  print(betterText)

  return float(totalText), float(betterText)

def parseLog(logPath):
  X = []
  bestY = []
  bestBetterY = []
  stopY = []
  stopBetterY = []
  incBetterY = []

  trainRound=None
  for l in open(logPath, 'r'):
    if l.startswith("- Round "):
      trainRound = parseRoundHeader(l)
      X.append(trainRound)
    elif "(best)" in l:
      if trainRound is None:
        print("Missed round header!!!")
        raise SystemExit
      # total, match, longerDer, shortedDer, betterScore = parseEvalLine(l)
      bestTotal, bestBetter = parseEvalLine(l)
      bestY.append(bestTotal)
      bestBetterY.append(bestBetter)
    elif "(STOP)" in l:
      if trainRound is None:
        print("Missed round header!!!")
        raise SystemExit
      # total, match, longerDer, shortedDer, betterScore = parseEvalLine(l)
      stopTotal, stopBetter = parseEvalLine(l)
      stopY.append(stopTotal)
      stopBetterY.append(stopBetter)
    elif "Incumbent" in l:
      if trainRound is None:
        print("Missed round header!!!")
        raise SystemExit
      # total, match, longerDer, shortedDer, betterScore = parseEvalLine(l)
      incTotal, incBetter = parseEvalLine(l)
      # stopY.append(stopTotal)
      incBetterY.append(incBetter)
  return X, bestY, bestBetterY, stopY, stopBetterY, incBetterY


X, bestY, bestBetterY, stopY, stopBetterY, incBetterY = parseLog(sys.argv[1])
pl.plot(X, bestY, 'b')
pl.plot(X, bestBetterY, 'c')
pl.plot(X, stopY, 'g')
pl.plot(X, stopBetterY, 'm')
pl.plot(X, incBetterY, 'r')
pl.show()
