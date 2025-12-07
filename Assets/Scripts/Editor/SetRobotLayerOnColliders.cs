using UnityEngine;
using UnityEditor;

public class SetRobotLayerOnColliders : EditorWindow
{
    [MenuItem("Tools/Set Robot Layer on so101 Colliders")]
    public static void SetLayerOnColliders()
    {
        GameObject so101 = GameObject.Find("so101");
        if (so101 == null)
        {
            Debug.LogError("Could not find so101 GameObject");
            return;
        }

        int robotLayer = LayerMask.NameToLayer("Robot");
        if (robotLayer == -1)
        {
            Debug.LogError("Robot layer does not exist");
            return;
        }

        int count = 0;
        Collider[] colliders = so101.GetComponentsInChildren<Collider>(true);
        
        foreach (Collider collider in colliders)
        {
            collider.gameObject.layer = robotLayer;
            count++;
        }

        Debug.Log($"Set {count} colliders to Robot layer");
    }
}