using UnityEngine;

public class OVRHandLayerInitializer
{
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void InitializeOVRHandLayers()
    {
        int interactiveLayer = LayerMask.NameToLayer("Interactive");
        
        if (interactiveLayer == -1)
        {
            Debug.LogError("Interactive layer does not exist! Please create it in the project settings.");
            return;
        }

        OVRHand[] ovrHands = Object.FindObjectsByType<OVRHand>(FindObjectsSortMode.None);
        
        if (ovrHands.Length == 0)
        {
            Debug.LogWarning("No OVRHand objects found in the scene.");
            return;
        }

        foreach (OVRHand hand in ovrHands)
        {
            SetLayerRecursively(hand.gameObject, interactiveLayer);
        }

        Debug.Log($"Set {ovrHands.Length} OVRHand objects and their children to Interactive layer (layer {interactiveLayer})");
    }

    private static void SetLayerRecursively(GameObject obj, int layer)
    {
        obj.layer = layer;
        
        foreach (Transform child in obj.transform)
        {
            SetLayerRecursively(child.gameObject, layer);
        }
    }
}