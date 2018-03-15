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

  return float(totalText), float(betterText)

def parseLog(logPath):
  taskName=None
  X = []
  bestY = []
  bestBetterY = []
  stopY = []
  stopBetterY = []

  incBetterX = [] # only track improvements for incumbent
  incBetterY = []

  lastIncumbent=0.0
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
      if incBetter <= lastIncumbent:
        continue
        
      lastIncumbent=incBetter
      incBetterX.append(trainRound)
      incBetterY.append(incBetter)
    elif l.startswith("name ="):
      taskName = l.split(" ")[2].strip()

  return taskName, X, bestY, bestBetterY, stopY, stopBetterY, incBetterX, incBetterY

# parse data
taskName, X, bestY, bestBetterY, stopY, stopBetterY, incBetterX, incBetterY = parseLog(sys.argv[1])

# build an artificial performance score
def buildPerfIndex(bestY, bestBetterY):
  return [best + better for best, better in zip(bestY, bestBetterY)]

def filterImprovements(X,Y):
  outX = [X[0]]
  outY = [Y[0]]
  bestY= outY[0]
  for x,y in zip(X,Y):
    if y > bestY:
      outX.append(x)
      outY.append(y)
      bestY = y
  return outX, outY


indexY = buildPerfIndex(bestY, bestBetterY)


### plot data ###
bestX=X[:len(bestY)]

# filter new incumbents
topX, topY = filterImprovements(bestX, bestY)

bestBetterX = X[:len(bestBetterY)]
topBetterX, topBetterY = filterImprovements(bestBetterX, bestBetterY)

stopBetterX = X[:len(stopBetterY)]
topStopX, topStopY = filterImprovements(stopBetterX, stopBetterY)

# plot training performance index
pl.plot(X[:len(indexY)], indexY, 'k:',label="perf. index")

# plot data

def plotAnnotated(rawX, Y, color, label):
  X = rawX[:len(Y)]
  topX, topY = filterImprovements(X, Y)
  pl.plot(topX, topY, '{}o'.format(color)) # best seen solution
  pl.plot(X, Y, color, label=label)

# current hit rate
plotAnnotated(X, bestY, "b", "opt: bestP")
plotAnnotated(X, stopY, "g", "opt: P@stop")

# fraction of improved programs
pl.plot(incBetterX[:len(incBetterY)], incBetterY, 'k.:',label="> ref (ceil)")

plotAnnotated(X, bestBetterY, "c", label="> ref (best P)")
plotAnnotated(X, stopBetterY, "m", label="> ref (P@stop)")

# extend incBetter line
if len(incBetterY) > 0:
  # append line w/o dot
  lastValue = incBetterY[-1]
  lastUpdateRound = incBetterX[-1]
  lastRound = X[-1]
  if lastUpdateRound != lastRound:
    pl.plot([lastUpdateRound, lastRound], [lastValue] * 2, 'k:')


# geknaupte axis
pl.plot([0, X[-1]], [1.0, 1.0], "k-")

pl.title("training on {} task".format(taskName))


### grid ###
# pl.rc('grid', linestyle=":", color='gray')
# pl.grid(True)


# axis
pl.xlabel("global step")
pl.ylabel("% of hold-out samples")
pl.legend()
pl.show()
