using UnityEngine;

public class SetCollidersToRobotLayer : MonoBehaviour
{
    void Start()
    {
        int robotLayer = LayerMask.NameToLayer("Robot");
        
        if (robotLayer == -1)
        {
            Debug.LogError("Robot layer does not exist!");
            return;
        }

        Collider[] colliders = GetComponentsInChildren<Collider>(true);
        
        foreach (Collider collider in colliders)
        {
            collider.gameObject.layer = robotLayer;
        }

        Debug.Log($"Set {colliders.Length} colliders to Robot layer (layer {robotLayer})");
    }
}