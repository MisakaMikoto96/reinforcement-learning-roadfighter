cw2(Q_test1,1)


function[weight] = cw2(Q_test1,choice)
%% Initialization 
weight = rand(4,5,3);
learningRate = 0.0001;
flag = 1; %determine whether to jump out of while loop
counter =1;
%decay_rate = 0.888888;
gammar = 1; %usually set as 1



%% ACTION CONSTANTS:
UP_LEFT = 1 ;
UP = 2 ;
UP_RIGHT = 3 ;



%% PROBLEM SPECIFICATION:

blockSize = 5 ; % This will function as the dimension of the road basis 
% images (blockSize x blockSize), as well as the view range, in rows of
% your car (including the current row).

n_MiniMapBlocksPerMap = 5 ; % determines the size of the test instance. 
% Test instances are essentially road bases stacked one on top of the
% other.

basisEpsisodeLength = blockSize - 1 ; % The agent moves forward at constant speed and
% the upper row of the map functions as a set of terminal states. So 5 rows
% -> 4 actions.

episodeLength = blockSize*n_MiniMapBlocksPerMap - 1 ;% Similarly for a complete
% scenario created from joining road basis grid maps in a line.

%discountFactor_gamma = 1 ; % if needed

rewards = [ 1, -1, -20 ] ; % the rewards are state-based. In order: paved 
% square, non-paved square, and car collision. Agents can occupy the same
% square as another car, and the collision does not end the instance, but
% there is a significant reward penalty.

probabilityOfUniformlyRandomDirectionTaken = 0.15 ; % Noisy driver actions.
% An action will not always have the desired effect. This is the
% probability that the selected action is ignored and the car uniformly 
% transitions into one of the above 3 states. If one of those states would 
% be outside the map, the next state will be the one above the current one.

roadBasisGridMaps = generateMiniMaps ; % Generates the 8 road basis grid 
% maps, complete with an initial location for your agent. (Also see the 
% GridMap class).



noCarOnRowProbability = 0.8 ; % the probability that there is no car 
% spawned for each row
seed = 1234;
rng(seed); % setting the seed for the random nunber generator

% Call this whenever starting a new episode:
MDP = generateMap( roadBasisGridMaps, n_MiniMapBlocksPerMap, blockSize, ...
    noCarOnRowProbability, probabilityOfUniformlyRandomDirectionTaken, ...
    rewards );


while flag
    %% GENERATE EPISODE
    %Set initialized vector for storing actions and rewards from the
    %episode
    actiontakens = ones(1,25); %+1 for td
    rewardtakens = zeros(1,24)
    stateFeatureFoueLines = zeros(4,5,25); % 4 rows after the current row

    
    currentTimeStep = 0 ;
    MDP = generateMap( roadBasisGridMaps, n_MiniMapBlocksPerMap, ...
        blockSize, noCarOnRowProbability, ...
        probabilityOfUniformlyRandomDirectionTaken, rewards );
    currentMap = MDP ;
    agentLocation = currentMap.Start ;
    startingLocation = agentLocation ; % Keeping record of initial location.
    
    % If you need to keep track of agent movement history:
    agentMovementHistory = zeros(episodeLength+1, 2) ;
    agentMovementHistory(currentTimeStep + 1, :) = agentLocation ;
        
    realAgentLocation = agentLocation ; % The location on the full test map.
    Return = 0;
    
    for i = 1:episodeLength
        
        % Use the $getStateFeatures$ function as below, in order to get the
        % feature description of a state:
        stateFeatures = MDP.getStateFeatures(realAgentLocation) % dimensions are 4rows x 5columns
        stateFeaturesFourLines(:,:,i) = stateFeatures;
        stateFeaturesFourLines(:,:,25) = zeros(4,5);

        for action = 1:3
            action_values(action) = ...
                sum ( sum( Q_test1(:,:,action) .* stateFeatures ) );
        end % for each possible action
        [~, actionTaken] = max(action_values)
               
        % The $GridMap$ functions $getTransitions$ and $getReward$ act as the
        % problems transition and reward function respectively.
        %
        % Your agent might not know these functions, but your simulator
        % does! (How wlse would we get our data?)
        %
        % $actionMoveAgent$ can be used to simulate agent (the car) behaviour.
        
        %     [ possibleTransitions, probabilityForEachTransition ] = ...
        %         MDP.getTransitions( realAgentLocation, actionTaken );
        %     [ numberOfPossibleNextStates, ~ ] = size(possibleTransitions);
        %     previousAgentLocation = realAgentLocation;
        
        [ agentRewardSignal, realAgentLocation, currentTimeStep, ...
            agentMovementHistory ] = ...
            actionMoveAgent( actionTaken, realAgentLocation, MDP, ...
            currentTimeStep, agentMovementHistory, ...
            probabilityOfUniformlyRandomDirectionTaken ) 
        
           % MDP.getReward( ...
           %          previousAgentLocation, realAgentLocation, actionTaken )
        
        
        % If you want to view the agents behaviour sequentially, and with a
        % moving view window, try using $pause(n)$ to pause the screen for $n$
        % seconds between each draw:
        
        [ viewableGridMap, agentLocation ] = setCurrentViewableGridMap( ...
            MDP, realAgentLocation, blockSize );
        % $agentLocation$ is the location on the viewable grid map for the
        % simulation. It is used by $refreshScreen$.
        
        currentMap = viewableGridMap; %#ok<NASGU>
        % $currentMap$ is keeping track of which part of the full test map
        % should be printed by $refreshScreen$ or $printAgentTrajectory$.
        
        
        %store every action and reward into initialized vectors
        actiontakens(i)=actionTaken;
        rewardtakens(i)=agentRewardSignal;
        
        Return = Return + agentRewardSignal;
    end
    
    %currentMap = MDP;
    %agentLocation = realAgentLocation;
    %Return;

%% Monte Carlo
    if choice ==0 % choose the Monte Carlo
        for i = 1:episodeLength   
            Return = sum(rewardtakens(i:24)); %target
            Q = sum(sum(weight(:,:,actiontakens(i)) .* stateFeaturesFourLines(:,:,i))) %predicted
            weight(:,:,actiontakens(i)) = ...
                    weight(:,:,actiontakens(i)) + ...
                    learningRate*(Return-Q) .* ...
                    stateFeaturesFourLines(:,:,i)
        end
        if max(abs(Return-Q))<0.0001 %whether it converge
            flag =0;
        end
        
        
%% TD       
    elseif choice ==1 %choose the TD-Learning
        for i = 1:episodeLength   
            Return = rewardtakens(i)...
                     +gammar*(sum(sum(weight(:,:,actiontakens(i+1))).*...
                     stateFeaturesFourLines(:,:,i+1))); %target
            Q = sum(sum(weight(:,:,actiontakens(i)).* stateFeaturesFourLines(:,:,i))); %predicted
            weight(:,:,actiontakens(i)) = ...
                weight(:,:,actiontakens(i))+...
                learningRate*(Return-Q).* ...
                stateFeaturesFourLines(:,:,i)
        end     
        if max(abs(Return-Q))<0.0001 %whether it converge
            flag = 0;
       
        end
    end
    counter=counter+1%counte the number of time
    %learningRate = learningRate + decay_rate

    
end
end
                
            

            
           
       
            
        